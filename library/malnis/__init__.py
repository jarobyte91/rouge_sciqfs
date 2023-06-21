from nltk.tokenize import sent_tokenize
import random
import numpy as np
from rouge import Rouge
from tqdm.auto import tqdm
from . import models

rouge = Rouge()

def show(x, n = 5):
    print(x.shape)
    return x.head(n = n)

def find_summary(
    query, 
    document, 
    starting_summary = None,
    metric = "rouge-1", 
    component = "f"
):
    sentences = sent_tokenize(document)
    summary = []
    if starting_summary:
        summary.append(starting_summary)
    current_score = 0.0
    while len(sentences) > 0:
        raw_scores = [
            rouge.get_scores(
                hyps = " ".join(summary + [c]), 
                refs = query
            )[0]
            for c in sentences
        ]
        scores = [d[metric][component] for d in raw_scores]
        max_score = max(scores)
        idx = np.argmax(scores)
        if max_score > current_score:
            summary.append(sentences[idx])
            current_score = max_score
            sentences.pop(idx)
        else:
            break
            
    return summary, max_score, raw_scores[idx]
