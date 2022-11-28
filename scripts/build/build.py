from multiprocessing import Pool
import argparse
import os
import malnis
# from malnis_dataset import show
import pandas as pd
# import importlib
from nltk.tokenize import sent_tokenize
from tqdm.auto import tqdm
from timeit import default_timer as t

def f(q, d, ss):
    summary, max_score, scores = malnis.find_summary(
        query = q,
        document = d,
        starting_summary = ss,
        metric = "rouge-2",
        component = "f"
    )
    return (
        q,
        d,
        summary,
        scores["rouge-1"]["f"],
        scores["rouge-2"]["f"],
        scores["rouge-l"]["f"]
    )


parser = argparse.ArgumentParser()
parser.add_argument("--full", action = "store_true", default = False)
parser.add_argument("--all-cpus", action = "store_true", default = False)
args = parser.parse_args()

full = args.full
all_cpus = args.all_cpus

print("reading data...", end = " ")
data = pd.read_csv("../../data/clean_examples.csv", index_col = 0)
print("done")
    
if all_cpus:
    n_cpus = os.cpu_count()
else:
#     n_cpus = os.cpu_count() - 1
    n_cpus = 5
print("cpus:", n_cpus)
    
if not full:
    data = data.head()

print("data shape:", data.shape)
# it = list()    
    
print("starting process...", end = " ")    
start = t()

with Pool(processes = n_cpus) as pool:
    results = pool.starmap(
        f,
        zip(data["query"], data.text, data.cited),
#         chunksize = 3
#         metric = "rouge-2",
#         component = "f"
    )

results = list(results)
    
time = (t() - start) / 60

print("done")
print("results:", len(results))
print(f"total time: {time: .2f} minutes")
print("building dataset...", end = " ")

# data = data\
# .assign(
#     summary = [s for s, m, d in results],
#     rouge_1 = [d["rouge-1"]["f"] for s, m, d in results],
#     rouge_2 = [d["rouge-2"]["f"] for s, m, d in results],
#     rouge_l = [d["rouge-l"]["f"] for s, m, d in results],
# )

data = pd.DataFrame(
    results,
    columns = ["query", "document", "summary", "r1", "r2", "rl"]
)
print("done")
print("data shape:", data.shape)
# print(data.head())
print("writing to disk...", end = " ")
data.to_pickle("results.pkl")
print("done")