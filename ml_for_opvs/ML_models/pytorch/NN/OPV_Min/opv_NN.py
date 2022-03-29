import copy
import math
import os
from argparse import ArgumentParser
from typing import Dict, List, Optional, Union

# for plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pkg_resources
import pytorch_lightning as pl
from pytorch_lightning.utilities import seed
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.optim import SGD, Adam
from opv_ml.ML_models.pytorch.OPV_Min.data.data import OPVDataModule
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.saving import save_hparams_to_yaml
from torch.utils import data
import wandb

from pytorch_lightning.loggers import WandbLogger

os.environ["WANDB_API_KEY"] = "95f67c3932649ca21ac76df3f88139dafacd965d"
os.environ["WANDB_MODE"] = "offline"


DATA_DIR = pkg_resources.resource_filename(
    "opv_ml", "data/process/OPV_Min/master_opv_ml_from_min.csv"
)

FRAG_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/train_frag_master.csv"
)

AUGMENT_SMILES_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/augmentation/train_aug_master15.csv"
)

BRICS_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/BRICS/master_brics_frag.csv"
)

MANUAL_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/manual_frag/master_manual_frag.csv"
)

FP_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/fingerprint/opv_fingerprint.csv"
)

CHECKPOINT_DIR = pkg_resources.resource_filename("opv_ml", "model_checkpoints/NN")

SEED_VAL = 4

# initialize weights for model
def initialize_weights(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight.data, gain=1)
        nn.init.constant_(model.bias.data, 0)


class NNModel(pl.LightningModule):
    def __init__(
        self,
        max_length,
        vocab_length,
        n_embedding,
        n_hidden,
        n_output,
        drop_prob,
        learning_rate,
    ):
        super().__init__()
        self.max_length = max_length
        self.vocab_length = vocab_length
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.drop_prob = drop_prob
        self.learning_rate = learning_rate
        self.embeds = nn.Embedding(vocab_length, n_embedding)
        self.linear1 = nn.Linear(n_embedding, n_output)
        self.linear2 = nn.Linear(max_length, n_output)
        self.loss = nn.MSELoss()
        self.dropout = nn.Dropout(drop_prob)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return Adam(
            self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8
        )

    def forward(self, x):
        embeds = self.embeds(x)
        out = self.dropout(embeds)
        out = self.linear1(out)
        out = out.squeeze()
        out = self.dropout(out)
        out = self.linear2(out)
        out = out.squeeze()
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        print(y_hat.size(), y.size())
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss, on_epoch=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--n_hidden", type=int, default=256)
        parser.add_argument("--n_embedding", type=int, default=128)
        parser.add_argument("--n_output", type=int, default=1)
        parser.add_argument("--drop_prob", type=float, default=0.3)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        return parser


def cli_main():
    pl.seed_everything(SEED_VAL)
    # ------------
    # wandb + sweep
    # ------------
    wandb_logger = WandbLogger(project="OPV_NN", offline=True, log_model=False)
    # online
    # wandb_logger = WandbLogger(project="OPV_NN")

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=-1)
    parser.add_argument("--accelerator", type=str, default="dp")
    # parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    # parser.add_argument("--data_dir", type=str, default=TRAIN_MASTER_DATA)
    # parser.add_argument("--data_dir", type=str, default=MANUAL_MASTER_DATA)
    parser.add_argument("--data_dir", type=str, default=FP_MASTER_DATA)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--dataloader_num_workers", type=int, default=3)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--logger", type=str, default=wandb_logger)
    parser = NNModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # ---------------
    # Data Conditions
    # ---------------
    unique_datatype = {
        "smiles": 1,
        "selfies": 0,
        "aug_smiles": 0,
        "hw_frag": 0,
        "aug_hw_frag": 0,
        "brics": 0,
        "manual": 0,
        "aug_manual": 0,
        "fingerprint": 0,
    }

    fp_radius = 3
    fp_nbits = 512

    # shuffle the PCE?
    shuffled = False

    if unique_datatype["smiles"] == 1:
        suffix = "/smi"
    elif unique_datatype["selfies"] == 1:
        suffix = "/selfies"
    elif unique_datatype["aug_smiles"] == 1:
        suffix = "/aug_smi"
    elif unique_datatype["hw_frag"] == 1:
        suffix = "/hw_frag"
    elif unique_datatype["aug_hw_frag"] == 1:
        suffix = "/aug_hw_frag"
    elif unique_datatype["brics"] == 1:
        suffix = "/brics"
    elif unique_datatype["manual"] == 1:
        suffix = "/manual"
    elif unique_datatype["aug_manual"] == 1:
        suffix = "/aug_manual"
    elif unique_datatype["fingerprint"] == 1:
        suffix = "/fp"

    if shuffled:
        suffix += "_shuffled"

    # parse arguments using the terminal shell (for ComputeCanada purposes)
    suffix = suffix + (
        "_NN-{epoch:02d}-{val_loss:.3f}"
        + "-n_hidden={}-n_embedding={}-drop_prob={}-lr={}-train_batch_size={}".format(
            args.n_hidden,
            args.n_embedding,
            args.drop_prob,
            args.learning_rate,
            args.train_batch_size,
        )
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", filename=CHECKPOINT_DIR + suffix, save_top_k=1, mode="min",
    )
    parser.add_argument("--callbacks", type=str, default=[checkpoint_callback])
    args = parser.parse_args()

    # pass args to wandb
    wandb.init(project="OPV_NN", config=args)
    config = wandb.config

    # ------------
    # data
    # ------------
    data_module = OPVDataModule(
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.dataloader_num_workers,
        smiles=unique_datatype["smiles"],
        selfies=unique_datatype["selfies"],
        aug_smiles=unique_datatype["aug_smiles"],
        hw_frag=unique_datatype["hw_frag"],
        aug_hw_frag=unique_datatype["aug_hw_frag"],
        brics=unique_datatype["brics"],
        manual=unique_datatype["manual"],
        aug_manual=unique_datatype["aug_manual"],
        fingerprint=unique_datatype["fingerprint"],
        fp_radius=fp_radius,
        fp_nbits=fp_nbits,
        pt_model=None,
        pt_tokenizer=None,
        shuffled=shuffled,
        seed_val=SEED_VAL,
    )
    data_module.setup()
    data_module.prepare_data()

    # ------------
    # model
    # ------------
    NN_model = NNModel(
        max_length=data_module.max_seq_length,
        vocab_length=data_module.vocab_length,
        n_embedding=args.n_embedding,
        n_hidden=args.n_hidden,
        n_output=args.n_output,
        drop_prob=args.drop_prob,
        learning_rate=args.learning_rate,
    )
    NN_model.apply(initialize_weights)

    # -----------
    # load trained model
    # -----------
    # best_model = LRModel.load_from_checkpoint(BEST_MODEL)

    # ------------
    # training
    # ------------
    # logger=wandb_logger
    trainer = pl.Trainer().from_argparse_args(args)
    trainer.fit(NN_model, data_module)
    trainer.test(NN_model, data_module)


if __name__ == "__main__":
    cli_main()
