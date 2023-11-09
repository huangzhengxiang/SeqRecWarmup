import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from options import args
from datasets import dataset_factory

args.dataset_code = "ml-20m" # 1 or 20
args.min_rating = 0
args.min_uc = 2
args.min_sc = 0
args.split = "leave_one_out"        

dataset = dataset_factory(args)
dataset.preprocess()

min_uc = args.min_uc
min_sc = args.min_sc
preprocessed_dataset_path = f'Data\preprocessed\{args.dataset_code}_min_rating{args.min_rating}-min_uc{min_uc}-min_sc{min_sc}-splitleave_one_out\dataset.pkl'

with open(preprocessed_dataset_path, 'rb') as f:
    data = pkl.load(f)

data = pd.DataFrame([[i, len(j)] for i, j in data['train'].items()], columns=['uid', 'seq_len'])

ticks = list(range(0, 400, 20)) + [np.inf]
labels = [f'{ticks[i]}' for i in range(len(ticks) - 1)]
cnts_1 = [len(data[(data['seq_len'] >= ticks[i]) & (data['seq_len'] < ticks[i+1])]) for i in range(len(ticks) - 1)]
print(cnts_1)

plt.figure(figsize=(12, 8))
plt.bar(labels, cnts_1, align='edge')
plt.xlabel('Seq Length')
plt.ylabel('Number of records')
plt.title(f'{args.dataset_code}')
plt.savefig(f'./Images/{args.dataset_code}_min_uc{min_uc}_min_sc{min_sc}_seqlen_distribution.png')
