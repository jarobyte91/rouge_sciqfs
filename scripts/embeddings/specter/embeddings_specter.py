#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.feature_extraction.text import TfidfVectorizer
from malnis import show
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm
# import scipy.sparse as sp
# from sklearn.linear_model import LogisticRegression 
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.decomposition import PCA, SparsePCA
# from sklearn.neural_network import MLPClassifier
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


# In[34]:


# The number of sentences in each paper goes from 59 to 4,447. I truncate to 512.

data = pd.read_pickle("../../data/sentence_labels.pkl")\
.reset_index(drop = True)\
.assign(n_sentences = lambda df: df.sentences.map(len))\
.assign(
    sentences = lambda df: df.sentences.map(lambda y: y[:512]),
    relevance = lambda df: df.relevance.map(lambda y: y[:512]),    
)

show(data)


# In[35]:


s = np.array(data.relevance.map(lambda x: [sum(x), len(x)]).tolist()).sum(axis = 0)
s[0] / s[1]


# In[36]:


data.n_sentences.describe()


# In[37]:


model = SentenceTransformer(
#     "sbert"
    "specter"
#     'all-MiniLM-L6-v2', 
#     cache_folder = "../assets"
#     "../cache/huggingface/transformers/"
#     cache_folder = "../cache/huggingface/transformers"
)
model.cuda()


# In[38]:


query_embeddings = model.encode(
    data["query"], 
#     show_progress_bar = True
)
print(query_embeddings.shape)


# In[39]:


dims = query_embeddings.shape[1]
sentence_embeddings = [
    model.encode(l)#.toarray() 
    for l in data.sentences
]
print(all([l.shape[1] == dims for l in sentence_embeddings]))

print(query_embeddings.shape[0] == len(sentence_embeddings))

print(sum([len(l) for l in data.sentences]))


# In[40]:


train = [
#     ((sp.csr_matrix(np.ones([l.shape[0],1])) * q) - l).power(2)
#     (q - l)**2
    np.concatenate([np.tile(q, (l.shape[0], 1)), l], axis = 1)
    for q, l in zip(query_embeddings, sentence_embeddings)
]
print(len(train))


# In[41]:


X = [torch.tensor(x) for x in train]
print(all([x.shape[0] <= 512 for x in X]))


# In[42]:


X = torch.nn.utils.rnn.pad_sequence(X, batch_first = True)
print(X.shape)


# In[43]:


# X = np.concatenate(train)#.toarray().T#.squeeze()
# print(X.shape)


# In[44]:


Y = torch.tensor([y for l in data.relevance for y in l])
Y.shape


# In[45]:


Y = torch.nn.utils.rnn.pad_sequence(
    [torch.tensor(x) for x in data.relevance], 
    batch_first = True
)\
.long()
Y.shape


# In[46]:


Y.sum()


# In[47]:


# (
#     X_train, 
#     X_devtest, 
#     Y_train, 
#     Y_devtest, 
#     relevance_train, 
#     relevance_devtest
# ) = train_test_split(
#     X, 
#     Y, 
#     data.relevance, 
#     random_state = 1,
#     test_size = 0.2
# )
# print("X_train", X_train.shape)
# print("X_devtest", X_devtest.shape)
# print("Y_train", Y_train.shape)
# print("Y_devtest", Y_devtest.shape)

# (
#     X_dev, 
#     X_test, 
#     Y_dev, 
#     Y_test, 
#     relevance_dev, 
#     relevance_test
# ) = train_test_split(
#     X_devtest, 
#     Y_devtest, 
#     relevance_devtest, 
#     random_state = 1,
#     test_size = 0.5
# )
# print("X_dev", X_dev.shape)
# print("X_test", X_test.shape)
# print("Y_dev", Y_dev.shape)
# print("Y_test", Y_test.shape)

folder = "/home/jarobyte/scratch/malnis_dataset/data/specter_embeddings/"

to_write = [
    ("X", X),
    ("Y", Y),
]

for name, contents in to_write:
    contents = contents.cpu().numpy()
    path = folder + name + ".npy"
    np.save(path, contents)
    print(name, "saved in", path)

