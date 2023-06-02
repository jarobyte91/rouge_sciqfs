print("Importing libraries...")

import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.utils.data as tud
from malnis.models import MLP
from random import randint
from sklearn.metrics import average_precision_score, roc_auc_score
from timeit import default_timer as timer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_id", type = str)
parser.add_argument("--full", action = "store_true", default = False)
# parser.add_argument("--random", action = "store_true", default = False)
# parser.add_argument("--epochs", type = int, default = 10)

args = parser.parse_args()
experiment_id = args.experiment_id
full = args.full
# random = args.random
# epochs = args.epochs

print("Loading data...")

data_folder = "/home/jarobyte/scratch/malnis_dataset/data/"

X_train = torch.tensor(np.load(data_folder + "embeddings/sbert/X_train.npy")).cuda()
Y_train = torch.tensor(np.load(data_folder + "embeddings/sbert/Y_train.npy")).cuda()

X_dev = torch.tensor(np.load(data_folder + "embeddings/sbert/X_dev.npy"))
Y_dev = torch.tensor(np.load(data_folder + "embeddings/sbert/Y_dev.npy"))

data_train = pd.read_pickle(data_folder + "data_train.pkl")
data_dev = pd.read_pickle(data_folder + "data_dev.pkl")

print("Creating model...")

# epochs = randint(10)
if full:
    epochs = 2000
    hidden_size = 50 * randint(2, 10)
    num_layers = randint(1, 4)
    weight_decay = 0
#     weight_decay = 10 ** -randint(2, 5)
else:
    epochs = 10
    hidden_size = 100
    num_layers = 2
    weight_decay = 0
    
# name = f"lstm_e{epochs}_h{hidden_size}_l{num_layers}"
name = experiment_id
        
clf = MLP(
    input_size = 768,
    hidden_size = hidden_size,
    num_layers = num_layers
)
clf.cuda()
clf.train()

print("Fitting model...")

start = timer()

train_log = clf.fit(
    X_train, 
    Y_train, 
    epochs = epochs, 
    weight_decay = weight_decay
)

training_minutes = (timer() - start) / 60

predictions = clf.predict(X_dev)

true_targets = np.concatenate(data_dev.relevance.to_list())

# reshaping the predictions to discard the padding
true_predictions = np.concatenate(
    [p[:len(l)] for p, l in zip(predictions, data_dev.relevance)]
)

train_data = train_log\
.assign(
    hidden_size = hidden_size,
    num_layers = num_layers,
    weight_decay = weight_decay,
    embeddings = "sbert",
    dev_average_precision = average_precision_score(true_targets, true_predictions),
    dev_roc_auc = roc_auc_score(true_targets, true_predictions),
    training_minutes = training_minutes
)

print(f"Training minutes: {training_minutes:.2f}")

output_folder = "/home/jarobyte/scratch/malnis_dataset/mlp/sbert/"
predictions_filename = output_folder + "predictions/" + name + ".npy"
train_log_filename = output_folder + "train_logs/" + name + ".pkl"
model_filename = output_folder + "models/" + name + ".pt"


print(f"Saving predictions at {predictions_filename}...")
np.save(predictions_filename, true_predictions)
print("Done!")

print(f"Saving train log at {train_log_filename}...")
train_data.to_pickle(train_log_filename)
print("Done!")

print(f"Saving model at {model_filename}...")
torch.save(clf.state_dict(), model_filename)
print("Done!")
