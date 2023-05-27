import pickle
import json
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
from model import RumourDetection

def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)


def freeze_params(model, freeze_layer_count=8):
    """Set requires_grad=False for each of model.parameters()"""
    modules = [model.embeddings, *model.encoder.layer[:freeze_layer_count]]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False




class LoggingCallback(Callback):
    @rank_zero_only
    def on_batch_end(self, trainer, pl_module):
        # lrs = pl_module.trainer.optimizers[0].param_groups[0]["lr"]
        pl_module.logger.log_metrics({"learning_rate": pl_module.trainer.optimizers[0].param_groups[0]["lr"]})

    @rank_zero_only
    def on_test_end(self, trainer, pl_module):
        print('Testing ends')
        print('--------------------------------------------')
        save_json(pl_module.metrics, pl_module.metrics_save_path)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        save_json(pl_module.metrics, pl_module.metrics_save_path)

    @rank_zero_only
    def on_init_start(self, trainer):
        print('Starting to init trainer!')
        print('--------------------------------------------')

    @rank_zero_only
    def on_init_end(self, trainer):
        print('trainer is init now')
        print('--------------------------------------------')

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        print('Training ends')
        print('--------------------------------------------')

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        print('Training starts')
        print('--------------------------------------------')

    @rank_zero_only
    def on_test_start(self, trainer, pl_module):
        print('Testing starts')
        print('--------------------------------------------')

import argparse
import random
import torch
from pathlib import Path
import pytorch_lightning as pl
# from utils import pickle_save
# from callback import LoggingCallback


def add_model_specific_args(parser):
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--num_workers", default=8, type=int, help="kwarg passed to DataLoader")
    parser.add_argument("--warmup_ratio", default=0.1, type=float, help="warm up rate steps")
    parser.add_argument("--save_top_k", type=int, default=1, required=False, help="How many checkpoints to save")
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--test_batch_size", default=32, type=int)

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--gradient_accumulation_steps",
        dest="accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_length",
        default=60,
        type=int,
        help="The maximum sentence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=-1,
        required=False,
        help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
    )
    parser.add_argument("--pre_encoder", default="bert-base-uncased", help="pre-trained encoder")
    parser.add_argument("--wandb", action="store_true", default=False, help="Whether to use wandb.")
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
    parser.add_argument("--n_val", type=int, default=-1, required=False, help="# examples. -1 means use all.")
    parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
    return parser


def train_model(model, args, logger):
    # init random seed
    pl.seed_everything(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # add model checkpoints callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir, filename="{f1_score:.4f}", monitor="f1_score", mode="max", save_top_k=args.save_top_k
    )

    # add logging callback
    logging_callback = ()
    # lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    extra_callbacks = [logging_callback, checkpoint_callback]
    # add early stop callback
    if args.early_stopping_patience > 0:
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor="f1_score",
            mode="max",
            patience=args.early_stopping_patience,
            verbose=True,
        )
        extra_callbacks.append(early_stopping_callback)

    if args.fp16:
        precision = 16
    else:
        precision = 32

    trainer = pl.Trainer.from_argparse_args(
        args,
        weights_summary="top",
        callbacks=extra_callbacks,
        logger=logger,
        precision=precision,
    )

    trainer.fit(model)

    return trainer


def main(args):
    odir = Path(args.output_dir)
    odir.mkdir(exist_ok=True)

    # init model
    model = RumourDetection(args)

    # init logger
    if args.wandb:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(name="twitter", project="NLP")
    else:
        logger = True

    trainer = train_model(model, args, True)

    # pickle_save(model.hparams, model.hparams_save_path)
    trainer.test(verbose=False, ckpt_path="best")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = add_model_specific_args(parser)
    args = parser.parse_args()
    main(args)

