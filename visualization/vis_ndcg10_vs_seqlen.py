import torch

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *

def vis_metrics_vs_seqlen(test_only=False, export_root=None):
    args.max_ticks=400
    args.ticks=10
    args.ticks = list(range(0, args.max_ticks, args.max_ticks//args.ticks)) + [np.inf]
    if test_only and export_root is None:
        raise NotADirectoryError()
    export_root = setup_train(args) if not test_only else export_root
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)

    metrics_types = [f'NDCG@{i}' for i in args.metric_ks] + [f'RECALL@{i}' for i in args.metrics_ks]
    metrics_set = {mtype: [] for mtype in metrics_types}
    for i in range(len(args.ticks) - 1):
        print(f"Testing ticks: {args.ticks[i]} , {args.ticks[i+1]}")
        avg_meters = trainer.test(i)
        for k, v in avg_meters:
            metrics_set[k].append(v)
    print(metrics_set)

if __name__ == '__main__':
    vis_metrics_vs_seqlen()