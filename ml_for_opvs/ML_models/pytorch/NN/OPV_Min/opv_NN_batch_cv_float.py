import copy
import math
import os
from argparse import ArgumentParser
from typing import Dict, List, Optional, Union

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
from ml_for_opvs.ML_models.pytorch.data.OPV_Min.data_cv import OPVDataModule
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.saving import save_hparams_to_yaml
from torch.utils import data
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import TrainingTypePlugin
from pytorch_lightning.accelerators import GPUAccelerator
from pytorch_lightning.plugins.precision import PrecisionPlugin

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from numpy import mean
from numpy import std

os.environ["WANDB_API_KEY"] = "95f67c3932649ca21ac76df3f88139dafacd965d"
os.environ["WANDB_MODE"] = "offline"


DATA_DIR = pkg_resources.resource_filename(
    "ml_for_opvs", "data/process/OPV_Min/master_ml_for_opvs_from_min.csv"
)

FRAG_MASTER_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/input_representation/OPV_Min/train_frag_master.csv"
)

AUGMENT_SMILES_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/input_representation/OPV_Min/augmentation/train_aug_master4.csv"
)

BRICS_MASTER_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/input_representation/OPV_Min/BRICS/master_brics_frag.csv"
)

MANUAL_MASTER_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/input_representation/OPV_Min/manual_frag/master_manual_frag.csv"
)

FP_MASTER_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/input_representation/OPV_Min/fingerprint/opv_fingerprint.csv"
)

CHECKPOINT_DIR = pkg_resources.resource_filename(
    "ml_for_opvs", "model_checkpoints/OPV_Min/NN"
)

SUMMARY_DIR = pkg_resources.resource_filename(
    "ml_for_opvs", "ML_models/pytorch/NN/OPV_Min/"
)

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
        self.embeds = nn.Linear(
            vocab_length, n_embedding
        )  # REMOVED EMBEDDINGS DUE TO FLOAT!
        self.linear1 = nn.Linear(max_length, n_output)
        self.loss = nn.MSELoss()
        self.dropout = nn.Dropout(drop_prob)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return Adam(
            self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8
        )

    def forward(self, x):
        x = x.float()
        out = self.dropout(x)
        out = self.linear1(out)
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
        print(y_hat.shape, y.shape)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        batch = self.all_gather(batch)
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y = y.cpu()
        y_hat = y_hat.cpu()
        corr_coef = np.corrcoef(y, y_hat)[0, 1]
        print("Y: ", y, "Y_HAT: ", y_hat, "CORR_COEF: ", corr_coef)
        self.log("test_loss", loss, on_epoch=True, sync_dist=True)
        self.log("test_corr_coef", corr_coef, on_epoch=True, sync_dist=True)

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
    SUMMARY_DIR = pkg_resources.resource_filename(
        "ml_for_opvs", "ML_models/pytorch/NN/OPV_Min/"
    )
    pl.seed_everything(SEED_VAL)
    # ------------
    # wandb + sweep
    # ------------
    # wandb_logger = WandbLogger(project="OPV_NN", offline=True, log_model=False)
    # online
    # wandb_logger = WandbLogger(project="OPV_NN")

    # log results
    summary_df = pd.DataFrame(
        columns=["Datatype", "R_mean", "R_std", "RMSE_mean", "RMSE_std", "num_of_data"]
    )

    # run batch of conditions
    unique_datatype = {
        "smiles": 0,
        "bigsmiles": 0,
        "selfies": 0,
        "aug_smiles": 0,
        "brics": 0,
        "manual": 0,
        "aug_manual": 0,
        "fingerprint": 0,
    }

    parameter_type = {
        "none": 1,
        "electronic": 0,
        "device": 0,
        "fabrication": 0,
    }

    for param in parameter_type:
        if parameter_type[param] == 1:
            dev_param = param
            if dev_param == "none":
                SUMMARY_DIR = SUMMARY_DIR + "none_opv_NN_results.csv"
            elif dev_param == "electronic":
                SUMMARY_DIR = SUMMARY_DIR + "electronic_opv_NN_results.csv"
            elif dev_param == "device":
                SUMMARY_DIR = SUMMARY_DIR + "device_opv_NN_results.csv"
            elif dev_param == "fabrication":
                SUMMARY_DIR = SUMMARY_DIR + "fabrication_opv_NN_results.csv"
    print(dev_param)

    for i in range(len(unique_datatype)):
        # ---------------
        # Data Conditions
        # ---------------
        parser = ArgumentParser()

        # ---------------
        # Custom accelerator and plugins
        # ---------------
        accelerator = GPUAccelerator(
            precision_plugin=PrecisionPlugin, training_type_plugin=TrainingTypePlugin
        )

        # ------------
        # args
        # ------------
        parser.add_argument("--gpus", type=int, default=-1)
        parser.add_argument("--accelerator", type=str, default="dp")
        parser.add_argument("--train_batch_size", type=int, default=128)
        parser.add_argument("--val_batch_size", type=int, default=64)
        parser.add_argument("--test_batch_size", type=int, default=64)
        parser.add_argument("--dataloader_num_workers", type=int, default=3)
        parser.add_argument("--max_epochs", type=int, default=500)
        parser.add_argument("--log_every_n_steps", type=int, default=100)
        parser.add_argument("--enable_progress_bar", type=bool, default=False)
        # parser.add_argument("--logger", type=str, default=wandb_logger)
        parser.add_argument("--logger", type=str, default=False)
        parser = NNModel.add_model_specific_args(parser)
        args = parser.parse_args()
        # reset conditions
        unique_datatype = {
            "smiles": 0,
            "bigsmiles": 0,
            "selfies": 0,
            "aug_smiles": 0,
            "brics": 0,
            "manual": 0,
            "aug_manual": 0,
            "fingerprint": 0,
        }
        index_list = list(np.zeros(len(unique_datatype) - 1))
        index_list.insert(i, 1)
        # set datatype with correct condition
        index = 0
        unique_var_keys = list(unique_datatype.keys())
        for j in index_list:
            unique_datatype[unique_var_keys[index]] = j
            index += 1

        fp_radius = 3
        fp_nbits = 512

        # shuffle the PCE?
        shuffled = False

        if unique_datatype["smiles"] == 1:
            suffix = "/smi"
            print("SMILES")
        elif unique_datatype["bigsmiles"] == 1:
            suffix = "/bigsmi"
            print("BigSMILES")
        elif unique_datatype["selfies"] == 1:
            suffix = "/selfies"
            print("SELFIES")
        elif unique_datatype["aug_smiles"] == 1:
            suffix = "/aug_smi"
            print("AUG_SMILES")
        elif unique_datatype["brics"] == 1:
            suffix = "/brics"
            print("BRICS")
        elif unique_datatype["manual"] == 1:
            suffix = "/manual"
            print("MANUAL")
        elif unique_datatype["aug_manual"] == 1:
            suffix = "/aug_manual"
            print("AUG_MANUAL")
        elif unique_datatype["fingerprint"] == 1:
            suffix = "/fp"
            print("FINGERPRINT: ", fp_radius, fp_nbits)

        if shuffled:
            suffix += "_shuffled"
            print("SHUFFLED")

        # parse arguments using the terminal shell (for ComputeCanada purposes)
        # suffix = suffix + (
        #     "_NN-{epoch:02d}-{val_loss:.3f}"
        #     + "-n_hidden={}-n_embedding={}-drop_prob={}-lr={}-train_batch_size={}".format(
        #         args.n_hidden,
        #         args.n_embedding,
        #         args.drop_prob,
        #         args.learning_rate,
        #         args.train_batch_size,
        #     )
        # )
        # checkpoint_callback = ModelCheckpoint(
        #     monitor="val_loss",
        #     filename=CHECKPOINT_DIR + suffix,
        #     save_top_k=1,
        #     mode="min",
        # )
        # parser.add_argument("--callbacks", type=str, default=[checkpoint_callback])
        # args = parser.parse_args()
        parser.add_argument("--callbacks", type=str, default=False)
        args = parser.parse_args()

        # pass args to wandb
        # wandb.init(project="OPV_NN", config=args)
        # config = wandb.config

        # ------------
        # cross-validation
        # ------------
        total_cv = 5
        cv_outer = list(range(0, total_cv))
        outer_corr_coef = list()
        outer_rmse = list()

        for cv in cv_outer:
            # ------------
            # data
            # ------------
            data_module = OPVDataModule(
                train_batch_size=args.train_batch_size,
                val_batch_size=args.val_batch_size,
                test_batch_size=args.test_batch_size,
                num_workers=args.dataloader_num_workers,
                smiles=unique_datatype["smiles"],
                bigsmiles=unique_datatype["bigsmiles"],
                selfies=unique_datatype["selfies"],
                aug_smiles=unique_datatype["aug_smiles"],
                brics=unique_datatype["brics"],
                manual=unique_datatype["manual"],
                aug_manual=unique_datatype["aug_manual"],
                fingerprint=unique_datatype["fingerprint"],
                fp_radius=fp_radius,
                fp_nbits=fp_nbits,
                cv=cv,
                pt_model=None,
                pt_tokenizer=None,
                shuffled=shuffled,
                seed_val=SEED_VAL,
            )
            data_module.setup()
            data_module.prepare_data(dev_param)

            # ------------
            # model
            # ------------
            print("LENGTHS: ", data_module.max_seq_length, data_module.vocab_length)
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

            # ------------
            # training
            # ------------
            trainer = pl.Trainer().from_argparse_args(args)
            trainer.fit(NN_model, data_module)

            # ------------
            # testing
            # ------------
            test_output = trainer.test(
                NN_model, data_module, ckpt_path=None, verbose=True
            )
            outer_corr_coef.append(test_output[0]["test_corr_coef"])
            outer_rmse.append(test_output[0]["test_loss"])

        # summarize KFold results
        print("R: %.3f (%.3f)" % (mean(outer_corr_coef), std(outer_corr_coef)))
        print("RMSE: %.3f (%.3f)" % (mean(outer_rmse), std(outer_rmse)))
        summary_series = pd.DataFrame(
            {
                "Datatype": suffix,
                "R_mean": mean(outer_corr_coef),
                "R_std": std(outer_corr_coef),
                "RMSE_mean": mean(outer_rmse),
                "RMSE_std": std(outer_rmse),
                "num_of_data": len(data_module.pce_train.dataset),
            },
            index=[0],
        )
        summary_df = pd.concat([summary_df, summary_series], ignore_index=True,)
    summary_df.to_csv(SUMMARY_DIR, index=False)


if __name__ == "__main__":
    cli_main()
