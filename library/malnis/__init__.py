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

# def find_summary(query, document):
#     sentences = sent_tokenize(document)
#     print("sentences", len(sentences))
#     current_score = 0.0
#     while len(sentences) > 0:
#         scores = [
#             rouge.get_scores(
# #             compute_score(
#                 " ".join([s for s in sentences if s != c]), 
#                 query
#             )[0]["rouge-1"]["f"]
#             for c in tqdm(sentences)
#         ]
#         max_score = max(scores)
# #         print(scores)
#         print(f"sentences: {len(sentences)}, score:{max_score:.3f}")
#         if max_score > current_score:
#             sentences.pop(np.argmax(scores))
#             current_score = max_score
#         else:
#             break
#     return sentences


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
#         print("sentences", len(sentences))
        raw_scores = [
            rouge.get_scores(
#             compute_score(
                hyps = " ".join(summary + [c]), 
                refs = query
            )[0]
            for c in sentences
        ]
        scores = [d[metric][component] for d in raw_scores]
        max_score = max(scores)
        idx = np.argmax(scores)
#         print("sesentences
#         print(max_score)
#         print(idx)
#         print(scores[idx])
#         print()
#         print(sentences[idx])
#         print()
#         print(scores)
#         print(f"summary: {len(summary)}, score:{max_score:.3f}")
        if max_score > current_score:
            summary.append(sentences[idx])
            current_score = max_score
            sentences.pop(idx)
#             print(summary)
        else:
            break
#         print()
#     print("max score:", max_score)
    return summary, max_score, raw_scores[idx]

# def compute_score(hypothesis, reference):
#     return random.uniform(0, 1)