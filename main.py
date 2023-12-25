import torch

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *
from visualization.vis_seqlen_distribution import vis_bucket


def main(test_only=False, export_root=None, backtrack=False):
    if test_only and export_root is None:
        raise NotADirectoryError()
    export_root = setup_train(args) if not test_only else export_root
    # 1. modify it to support multiple test_loaders.
    # 2. split the test_dataset.
    train_loader, val_loader, test_loader, mask_token, pkl_path = dataloader_factory(args)
    model = model_factory(args)
    model.set_mask_token(mask_token)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root, backtrack)
    if not test_only:
        trainer.train()

    test_model = (input('Test model with test dataset? y/[n]: ') == 'y')

    def trainer_inf(args, trainer, metric="NDCG@10"):
        meter_list = []
        for j in range(len(args.ticks)-1):
            avg_meters = trainer.test(j)
            if metric in avg_meters:
                meter_list.append(avg_meters[metric])
            else:
                meter_list.append(0.)
        return meter_list
    
    if test_model:
        # 3. In trainer.test test each test_loader
        meter_list_backtrack = trainer_inf(args, trainer)
        if backtrack:
            trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root, False)
            vis_bucket(args, 
                       lines={"backtrack": meter_list_backtrack,
                              "original":trainer_inf(args, trainer)},
                       pkl_path=pkl_path)
        else:
            vis_bucket(args, lines={"original":meter_list_backtrack},pkl_path=pkl_path)


if __name__ == '__main__':
    if args.mode == 'train':
        main()
    elif args.mode == "test":
        if args.dataset_code == "ml-1m":
            main(True, os.path.join(".","experiments","ml-1m-best"), args.bert_backtrack)
        elif args.dataset_code == "ml-20m":
            main(True, os.path.join(".","experiments","ml-20m-best"), args.bert_backtrack)
        else:
            raise NotImplementedError
    else:
        raise ValueError('Invalid mode')
