print("Loading libraries...")
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')
from sklearn.feature_extraction.text import TfidfVectorizer
# from tqdm.notebook import tqdm
from sklearn.neural_network import MLPClassifier
# from nltk.tokenize import sent_tokenize
# from rouge import Rouge
# from sklearn.model_selection import train_test_split
# from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score, average_precision_score
# PrecisionRecallDisplay, RocCurveDisplay
# from malnis import show
import pickle
import argparse
from random import randint
from timeit import default_timer as timer

print("Parsing arguments...")
parser = argparse.ArgumentParser()
parser.add_argument("--experiment_id", type = str)
parser.add_argument("--full", action = "store_true", default = False)
# parser.add_argument("--random", action = "store_true", default = False)
# parser.add_argument("--epochs", type = int, default = 10)

args = parser.parse_args()
experiment_id = args.experiment_id
full = args.full
if not full:
    n = 100
    total_epochs = 10
else:
    n = 3000000
    total_epochs = 16

args = parser.parse_args()
experiment_id = args.experiment_id
full = args.full
# random = args.random
# epochs = args.epochs

print("Reading data...")
data_folder = "/home/jarobyte/scratch/malnis_dataset/data/"
train = pd.read_pickle(data_folder + "data_train.pkl")
dev = pd.read_pickle(data_folder + "data_dev.pkl")
train_targets = np.concatenate(train.relevance.to_list())

print("Fitting vectorizer...")
corpus = train.sentences.sum()
vectorizer = TfidfVectorizer(
    analyzer = "char",
    ngram_range = (3, 3)
)
print("Vectorizer:", vectorizer)

print("Examples:", n)
train_features = vectorizer.fit_transform(corpus[:n])
print("train_features:", train_features.shape)

print("Fitting classfier...")
X = train_features
Y = train_targets[:n]

hidden_size = randint(1, 4) * 100
num_layers = randint(1, 4)
hidden_layer_sizes = [hidden_size for n in range(num_layers)]
clf = MLPClassifier(
    hidden_layer_sizes = hidden_layer_sizes,
    verbose = True,
    max_iter = total_epochs
)
print(clf)
# clf.partial_fit(X, Y, classes = np.unique(Y))
start = timer()
print("Start:", start)
clf.fit(X, Y)
end = timer() - start
print("End:", end)


# predicting on dev set
print("Predicting on dev set...")
dev_features = [vectorizer.transform(l) for l in dev.sentences]
dev_preds = [clf.predict_proba(f) for f in dev_features]
dev_targets = np.concatenate(dev.relevance.to_list())
dev_preds_flat = np.concatenate(dev_preds)[:, 1]
average_precision = average_precision_score(dev_targets, dev_preds_flat)
roc_auc = roc_auc_score(dev_targets, dev_preds_flat)
print("Average Precision:", average_precision)
print("ROC AUC:", roc_auc)
train_log = pd.DataFrame(
    list(enumerate(clf.loss_curve_, 1)),
    columns = ["epoch", "train_loss"]
)\
.assign(
    features = "chars",
    hidden_size = hidden_size,
    num_layers = num_layers,
    total_epochs = total_epochs,
    training_minutes = end / 60,
    dev_average_precision = average_precision,
    dev_roc_auc = roc_auc
)

print("Saving to disk...")
path = "/home/jarobyte/scratch/malnis_dataset/mlp_tfidf/chars/"
model_path = path + f"models/{experiment_id}.pkl"
predictions_path = path + f"predictions/{experiment_id}.npy"
train_log_path = path + f"train_logs/{experiment_id}.pkl"

with open(model_path, "wb") as file:
    pickle.dump(clf, file)
print("Model saved in:", model_path)

np.save(predictions_path, dev_preds_flat)
print("Predictions saved in:", predictions_path)

train_log.to_pickle(train_log_path)
print("Train log saved in:", train_log_path)
    
# with open("/home/jarobyte/scratch/malnis_dataset/words/predictions/test.pkl", "rb") as file:
#     clf_2 = pickle.load(file)
#     
# dev_preds = [clf_2.predict_proba(f) for f in tqdm(dev_features)]
# dev_preds_flat = np.concatenate(dev_preds)[:, 1]
# dev_preds_flat.shape
# average_precision_score(dev_targets, dev_preds_flat)
# roc_auc_score(dev_targets, dev_preds_flat)
# 
