print("Importing libraries...")

import pandas as pd
import matplotlib.pyplot as plt
from malnis import show
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm
from sklearn.metrics import log_loss, PrecisionRecallDisplay, RocCurveDisplay
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import pytorch_lightning as pl
import torch.utils.data as tud
import seaborn as sns
sns.set()

print("Loading data...")

data_folder = "/home/jarobyte/scratch/malnis_dataset/data/"

data_train = pd.read_pickle(data_folder + "data_train.pkl")\
.reset_index(drop = True)\
.assign(n_sentences = lambda df: df.sentences.map(len))\
.assign(
    sentences = lambda df: df.sentences.map(lambda y: y[:512]),
    relevance = lambda df: df.relevance.map(lambda y: y[:512]),    
)#.head()

data_dev = pd.read_pickle(data_folder + "data_dev.pkl")\
.reset_index(drop = True)\
.assign(n_sentences = lambda df: df.sentences.map(len))\
.assign(
    sentences = lambda df: df.sentences.map(lambda y: y[:512]),
    relevance = lambda df: df.relevance.map(lambda y: y[:512]),    
)#.head()

data_test = pd.read_pickle(data_folder + "data_test.pkl")\
.reset_index(drop = True)\
.assign(n_sentences = lambda df: df.sentences.map(len))\
.assign(
    sentences = lambda df: df.sentences.map(lambda y: y[:512]),
    relevance = lambda df: df.relevance.map(lambda y: y[:512]),    
)#.head()

print("Loading model...")

model = SentenceTransformer(
    "sbert"
)
model.cuda()

output_folder = data_folder + "embeddings/sbert/"

def compute_embeddings(partition, data):
    print("Computing embeddings for:", partition)
    
    query_embeddings = model.encode(
        data["query"], 
    )
    print("query embeddings", query_embeddings.shape)

    dims = query_embeddings.shape[1]
    sentence_embeddings = [
        model.encode(l)#.toarray() 
        for l in data.sentences
    ]
    
    print("all the sentence embeddings of the right dim:", all([l.shape[1] == dims for l in sentence_embeddings]))
    print("query and sentence embeddings of the same length:", query_embeddings.shape[0] == len(sentence_embeddings))
    print("sum of lenghts of data sentences:", sum([len(l) for l in data.sentences]))

    print("Broadcasting...")
    train = [
    #     ((sp.csr_matrix(np.ones([l.shape[0],1])) * q) - l).power(2)
    #     (q - l)**2
        np.concatenate([np.tile(q, (l.shape[0], 1)), l], axis = 1)
        for q, l in zip(query_embeddings, sentence_embeddings)
    ]
    print(len(train))

    print("Creating tensors...")
    X = [torch.tensor(x) for x in train]
    print(all([x.shape[0] <= 512 for x in X]))

    X = torch.nn.utils.rnn.pad_sequence(X, batch_first = True)
    print(X.shape)

    Y = torch.tensor([y for l in data.relevance for y in l])
    print(Y.shape)

    Y = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x) for x in data.relevance], 
        batch_first = True
    )\
    .long()
    print(Y.shape)
    
    print("Writing to disk...")
    to_write = [
        (f"X_{partition}", X),
        (f"Y_{partition}", Y),
    ]

    for name, contents in to_write:
        contents = contents.cpu().numpy()
        path = f"{output_folder}{name}.npy"
        np.save(path, contents)
        print(name, "saved in", path)

compute_embeddings("train", data_train)
compute_embeddings("dev", data_dev)
compute_embeddings("test", data_test)