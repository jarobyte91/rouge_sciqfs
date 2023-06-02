# from multiprocessing import Pool
import argparse
# import os
import malnis
# from malnis_dataset import show
import pandas as pd
# import importlib
from nltk.tokenize import sent_tokenize
# from tqdm.auto import tqdm
from timeit import default_timer as timer

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
parser.add_argument("array_id", type = int)
parser.add_argument("--full", action = "store_true", default = False)
# parser.add_argument("--all-cpus", action = "store_true", default = False)
args = parser.parse_args()

array_id = args.array_id
full = args.full
# all_cpus = args.all_cpus

# print("array_id", array_id)

print("reading data...", end = " ")
data = pd.read_csv("../../data/clean_examples.csv", index_col = 0)
print("done")
print("full data shape:", data.shape)

a = (array_id - 1) * 100
b = array_id * 100
data = data.iloc[a:b, :]
print("data indices", a, b) 

# if all_cpus:
#     n_cpus = 64
# else:
# #     n_cpus = os.cpu_count() - 1
#     n_cpus = 4
# print("cpus:", n_cpus)

if not full:
    data = data.head(2)

print("actual data shape:", data.shape)
# it = list()    
    
print("starting process...", end = " ")    
start = timer()

# with Pool(processes = n_cpus) as pool:
#     results = pool.starmap(
#         f,
#         zip(data["query"], data.text, data.cited),
# #         chunksize = 3
# #         metric = "rouge-2",
# #         component = "f"
#     )

# results = list(results)


# last working
# results = [
#     f(q, t, c) 
#     for q, t, c in zip(
#         data["query"], 
#         data.text, 
#         data.cited
#     )
# ]
    
results = []
for q, t, c in zip(data["query"], data.text, data.cited):
    results.append(f(q, t, c))
    data = pd.DataFrame(
        results,
        columns = ["query", "document", "summary", "r1", "r2", "rl"]
    )
    data.to_pickle(f"temp/results_{array_id}.pkl")

time = (timer() - start) / 60

print("done")
print("results:", len(results))
print(f"total time: {time: .2f} minutes")

print("building dataset...", end = " ")
data = pd.DataFrame(
    results,
    columns = ["query", "document", "summary", "r1", "r2", "rl"]
)
print("done")
print("data shape:", data.shape)

# print(data.head())
print("writing to disk...", end = " ")
data.to_pickle(f"temp/results_{array_id}.pkl")
print("done")