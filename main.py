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
    train_loader, val_loader, test_loader, mask_token = dataloader_factory(args)
    model = model_factory(args)
    model.set_mask_token(mask_token)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root, backtrack)
    if not test_only:
        trainer.train()

    test_model = (input('Test model with test dataset? y/[n]: ') == 'y')
    if test_model:
        # 3. In trainer.test test each test_loader
        for j in range(len(args.ticks)-1):
            print(args.ticks[j],args.ticks[j+1])
            trainer.test(j)
    vis_bucket(args, )


if __name__ == '__main__':
    if args.mode == 'train':
        main()
    elif args.mode == "test":
        main(True, os.path.join(".","experiments","test_2023-11-17_6"), args.bert_backtrack)
    else:
        raise ValueError('Invalid mode')
