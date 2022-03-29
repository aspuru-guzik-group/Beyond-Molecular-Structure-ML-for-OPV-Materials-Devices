import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pkg_resources
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.optim import SGD, Adam
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.saving import save_hparams_to_yaml
from torch.utils import data

from opv_ml.ML_models.pytorch.data.OPV_Min.data import OPVDataModule

DATA_DIR = pkg_resources.resource_filename(
    "opv_ml", "data/process/master_opv_ml_from_min.csv"
)

TRAIN_FRAG_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/train_frag_master.csv"
)

TRAIN_AUG_SMI_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/augmentation/train_aug_master15.csv"
)

MANUAL_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/manual_frag/master_manual_frag.csv"
)

BRICS_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/BRICS/master_brics_frag.csv"
)

FP_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/fingerprint/opv_fingerprint.csv"
)

CHECKPOINT_DIR = pkg_resources.resource_filename("opv_ml", "model_checkpoints/LSTM")

os.environ["WANDB_API_KEY"] = "95f67c3932649ca21ac76df3f88139dafacd965d"
os.environ["WANDB_MODE"] = "offline"

SEED_VAL = 4

# initialize weights for model
def initialize_weights(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight.data, gain=1)
        nn.init.constant_(model.bias.data, 0)


class LSTMModel(pl.LightningModule):
    def __init__(
        self,
        dataset_size,
        output_size,
        n_embedding,
        n_hidden,
        n_layers,
        learning_rate,
        drop_prob,
        direction_bool=False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.direction_bool = direction_bool
        self.learning_rate = learning_rate
        self.embeds = nn.Embedding(dataset_size, n_embedding)
        self.lstm = nn.LSTM(
            n_embedding,
            n_hidden,
            n_layers,
            dropout=drop_prob,
            batch_first=True,
            bidirectional=direction_bool,
        )
        self.dropout = nn.Dropout(drop_prob)
        self.loss = nn.MSELoss()
        if direction_bool:
            self.linear = nn.Linear(n_hidden * 2, output_size)
        else:
            self.linear = nn.Linear(n_hidden, output_size)

        self.save_hyperparameters()

    def configure_optimizers(self):
        return Adam(
            self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8
        )

    def forward(self, x):
        # initialize hidden and cell state
        if self.direction_bool:
            h_0 = torch.zeros(
                2 * self.n_layers, x.size(0), self.n_hidden
            ).requires_grad_()
            c_0 = torch.zeros(
                2 * self.n_layers, x.size(0), self.n_hidden
            ).requires_grad_()
        else:
            h_0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden).requires_grad_()
            c_0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden).requires_grad_()
        # initialize hidden and cell states with normal, orthogonal, and xavier
        torch.nn.init.normal_(h_0, 0, 1)
        torch.nn.init.normal_(c_0, 0, 1)

        x = x.view(x.size(0), -1)
        x = x.long()
        h_0, c_0 = h_0.to(self.device), c_0.to(self.device)
        # h_0, c_0 = h_0.type_as(x), c_0.type_as(x)
        embeds = self.embeds(x)
        lstm_out, (h_0, c_0) = self.lstm(embeds, (h_0, c_0))
        if self.direction_bool:
            h_0 = h_0.contiguous().view(-1, 2 * self.n_hidden)
        else:
            h_0 = h_0.contiguous().view(-1, self.n_hidden)
        out = self.dropout(h_0)
        # TODO: add layer normalization, look at weights if theyre big or not - weight decay,
        out = self.linear(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.reshape(-1)
        # print(y_hat, y_hat.size(), y)
        loss = self.loss(y_hat, y)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.reshape(-1)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)
        # corr_coef = np.corrcoef(y_hat.cpu(), y.cpu())[0, 1]
        # coef_determination = corr_coef ** 2
        # self.log("val_loss_r2", coef_determination, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.reshape(-1)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss, on_epoch=True)


def cli_main():
    pl.seed_everything(SEED_VAL)

    # ------------
    # wandb + sweep
    # ------------
    # wandb_logger = WandbLogger(project="OPV_LSTM", log_model=False, offline=False)
    # online
    # wandb_logger = WandbLogger(project="OPV_LSTM")

    # ------------
    # checkpoint + hyperparameter manual tuning
    # ------------

    n_hidden = 256
    n_embedding = 128
    drop_prob = 0.3
    learning_rate = 1e-2
    train_batch_size = 128

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--n_hidden", type=int, default=n_hidden)
    parser.add_argument("--n_embedding", type=int, default=n_embedding)
    parser.add_argument("--n_output", type=int, default=1)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--drop_prob", type=float, default=drop_prob)
    parser.add_argument("--learning_rate", type=float, default=learning_rate)
    parser.add_argument("--direction_bool", type=bool, default=False)
    parser.add_argument("--train_batch_size", type=int, default=train_batch_size)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=-1)
    parser.add_argument("--accelerator", type=str, default="dp")
    parser.add_argument("--dataloader_num_workers", type=int, default=3)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    # parser.add_argument("--logger", type=str, default=wandb_logger)
    args = parser.parse_args()

    # ---------------
    # Data Conditions
    # ---------------
    # smiles = True
    smiles = False
    # aug_smiles = True
    aug_smiles = False  # change to aug_smi (in suffix)
    aug_max = False  # change to aug_frag (in suffix)
    aug_pairs = False  # change to aug_pair_frag (in suffix)
    brics = False
    data_aug = False  # change to aug (in suffix)
    manual = False
    # manual = True
    fingerprint = True
    # fingerprint = False
    # shuffled = True
    shuffled = False

    fp_radius = 2
    fp_nbits = 512

    # 0 - SMILES, 2 - SELFIES
    input_rep = 0
    # input_rep = 2

    if aug_smiles:
        suffix = "/aug_smi"
    elif smiles and input_rep == 0:
        suffix = "/smi"
    elif smiles and input_rep == 2:
        suffix = "/selfies"
    elif aug_max and manual:
        suffix = "/aug_manual_frag"
    elif aug_max:
        suffix = "/aug_frag"
    elif aug_pairs:
        suffix = "/aug_frag_pairs"
    elif brics:
        suffix = "/brics"
    elif manual:
        suffix = "/manual"
    elif fingerprint:
        suffix = "/fp_r=" + str(fp_radius) + "_nbits" + str(fp_nbits)
    else:
        suffix = "/frag"

    if shuffled:
        suffix += "_shuffled"

    # parse arguments using the terminal shell (for ComputeCanada purposes)
    suffix = suffix + (
        "_LSTM-{epoch:02d}-{val_loss:.3f}"
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
    wandb.init(project="OPV_LSTM", config=args)
    config = wandb.config

    # ------------
    # data
    # ------------
    data_module = OPVDataModule(
        data_dir=args.data_dir,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.dataloader_num_workers,
        input=input_rep,  # 0 - SMILES, 2 - SELFIES
        smiles=smiles,
        data_aug=data_aug,  # change to aug (in suffix)
        aug_smiles=aug_smiles,  # change to aug_smi (in suffix)
        aug_max=aug_max,  # change to aug_frag (in suffix)
        aug_pairs=aug_pairs,  # change to aug_pair_frag (in suffix)
        brics=brics,
        manual=manual,
        fingerprint=fingerprint,
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
    lstm_model = LSTMModel(
        dataset_size=data_module.data_size,
        output_size=1,
        n_embedding=args.n_embedding,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        learning_rate=args.learning_rate,
        drop_prob=args.drop_prob,
        direction_bool=args.direction_bool,
    )
    lstm_model.apply(initialize_weights)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer().from_argparse_args(args)
    trainer.fit(lstm_model, data_module)


if __name__ == "__main__":
    cli_main()
