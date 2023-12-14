import os
import sys
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from options import args
from datasets import dataset_factory

def vis_bucket(args,
                preprocessed_root=os.path.join(".", "Data", "preprocessed"),
                figure_root=os.path.join(".","MyImages"),
                lines=None):
    os.makedirs(preprocessed_root, exist_ok=True)
    os.makedirs(figure_root, exist_ok=True)
    
    dataset = dataset_factory(args)
    dataset.preprocess()

    preprocessed_dataset_path = os.path.join(preprocessed_root,f'{args.dataset_code}_min_rating{args.min_rating}-min_uc{args.min_uc}-min_sc{args.min_sc}-split{args.split}\dataset.pkl')

    ticks = args.ticks

    with open(preprocessed_dataset_path, 'rb') as f:
        data = pkl.load(f)

    print(f"user count: {len(data['umap'])}")
    print(f"item count: {len(data['smap'])}")

    data = pd.DataFrame([[i, len(j)] for i, j in data['train'].items()], columns=['uid', 'seq_len'])

    labels = [f'{ticks[i]}' for i in range(len(ticks) - 1)]
    cnts_1 = [len(data[(data['seq_len'] >= ticks[i]) & (data['seq_len'] < ticks[i+1])]) for i in range(len(ticks) - 1)]
    print(f"number of users: {cnts_1}")

    plt.figure(figsize=(12, 8))
    plt.bar(labels, cnts_1, align='edge')
    if not (lines is None):
        for line in lines:
            plt.plot(labels, line)
    plt.xlabel('Seq Length')
    plt.ylabel('Number of users')
    plt.title(f'{args.dataset_code}')
    plt.savefig(os.path.join(figure_root, f'{args.dataset_code}_min_rating{args.min_rating}_min_uc{args.min_uc}_min_sc{args.min_sc}_split{args.split}_seqlen_distribution.png'))


if __name__=="__main__":
    args.dataset_code = "ml-20m"
    args.min_rating = 4
    args.min_uc = 2
    args.min_sc = 0
    args.split = "leave_one_out"       
    
    vis_bucket(args, os.path.join(".", "Data", "preprocessed"), os.path.join(".","MyImages")) 