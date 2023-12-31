from .bert import BERTTrainer
from .dae import DAETrainer
from .vae import VAETrainer


TRAINERS = {
    BERTTrainer.code(): BERTTrainer,
    DAETrainer.code(): DAETrainer,
    VAETrainer.code(): VAETrainer
}


def trainer_factory(args, model, train_loader, val_loader, test_loader, export_root, backtrack=False):
    trainer = TRAINERS[args.trainer_code]
    if backtrack:
        return trainer(args, model, train_loader, val_loader, test_loader, export_root, backtrack)
    else:
        return trainer(args, model, train_loader, val_loader, test_loader, export_root)
