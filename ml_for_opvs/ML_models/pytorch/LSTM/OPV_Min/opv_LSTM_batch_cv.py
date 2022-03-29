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

from opv_ml.ML_models.pytorch.data.OPV_Min.data_cv import OPVDataModule

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from numpy import mean
from numpy import std

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

CHECKPOINT_DIR = pkg_resources.resource_filename(
    "opv_ml", "model_checkpoints/OPV_Min/LSTM"
)

SUMMARY_DIR = pkg_resources.resource_filename(
    "opv_ml", "ML_models/pytorch/LSTM/OPV_Min/opv_LSTM_batch_cv.csv"
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
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        # corr_coef = np.corrcoef(y_hat.cpu(), y.cpu())[0, 1]
        # coef_determination = corr_coef ** 2
        # self.log("val_loss_r2", coef_determination, on_epoch=True)

    def test_step(self, batch, batch_idx):
        batch = self.all_gather(batch)
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.reshape(-1)
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
        parser.add_argument("--n_hidden", type=int, default=128)
        parser.add_argument("--n_embedding", type=int, default=128)
        parser.add_argument("--n_output", type=int, default=1)
        parser.add_argument("--n_layers", type=int, default=1)
        parser.add_argument("--drop_prob", type=float, default=0.3)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--direction_bool", type=bool, default=False)
        return parser


def cli_main():
    pl.seed_everything(SEED_VAL)

    # ------------
    # wandb + sweep
    # ------------
    # wandb_logger = WandbLogger(project="OPV_LSTM", log_model=False, offline=False)
    # online
    # wandb_logger = WandbLogger(project="OPV_LSTM")

    # log results
    summary_df = pd.DataFrame(
        columns=["Datatype", "R_mean", "R_std", "RMSE_mean", "RMSE_std"]
    )

    # run batch of conditions
    unique_datatype = {
        "smiles": 0,
        "bigsmiles": 0,
        "selfies": 0,
        "aug_smiles": 0,
        "hw_frag": 0,
        "aug_hw_frag": 0,
        "brics": 0,
        "manual": 0,
        "aug_manual": 0,
        "fingerprint": 0,
    }
    for i in range(len(unique_datatype)):
        # ---------------
        # Data Conditions
        # ---------------
        parser = ArgumentParser()
        # ------------
        # args
        # ------------
        parser.add_argument("--gpus", type=int, default=1)
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
        parser = LSTMModel.add_model_specific_args(parser)
        args = parser.parse_args()
        # reset conditions
        unique_datatype = {
            "smiles": 0,
            "bigsmiles": 0,
            "selfies": 0,
            "aug_smiles": 0,
            "hw_frag": 0,
            "aug_hw_frag": 0,
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
        elif unique_datatype["bigsmiles"] == 1:
            suffix = "/bigsmi"
            break
        elif unique_datatype["selfies"] == 1:
            suffix = "/selfies"
            break
        elif unique_datatype["aug_smiles"] == 1:
            suffix = "/aug_smi"
        elif unique_datatype["hw_frag"] == 1:
            suffix = "/hw_frag"
            break
        elif unique_datatype["aug_hw_frag"] == 1:
            suffix = "/aug_hw_frag"
            break
        elif unique_datatype["brics"] == 1:
            suffix = "/brics"
            break
        elif unique_datatype["manual"] == 1:
            suffix = "/manual"
        elif unique_datatype["aug_manual"] == 1:
            suffix = "/aug_manual"
        elif unique_datatype["fingerprint"] == 1:
            suffix = "/fp"
            break

        if shuffled:
            suffix += "_shuffled"

        # parse arguments using the terminal shell (for ComputeCanada purposes)
        # suffix = suffix + (
        #     "_LSTM-{epoch:02d}-{val_loss:.3f}"
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

        # pass args to wandb
        # wandb.init(project="OPV_LSTM", config=args)
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
                hw_frag=unique_datatype["hw_frag"],
                aug_hw_frag=unique_datatype["aug_hw_frag"],
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

            # ------------
            # testing
            # ------------
            test_output = trainer.test(
                lstm_model, data_module, ckpt_path=None, verbose=True
            )
            outer_corr_coef.append(test_output[0]["test_corr_coef"])
            outer_rmse.append(test_output[0]["test_loss"])

        # summarize KFold results
        print("R: %.3f (%.3f)" % (mean(outer_corr_coef), std(outer_corr_coef)))
        print("RMSE: %.3f (%.3f)" % (mean(outer_rmse), std(outer_rmse)))
        summary_series = pd.Series(
            [
                suffix,
                mean(outer_corr_coef),
                std(outer_corr_coef),
                mean(outer_rmse),
                std(outer_rmse),
            ],
            index=summary_df.columns,
        )
        summary_df = summary_df.append(summary_series, ignore_index=True)
    summary_df.to_csv(SUMMARY_DIR, index=False)


if __name__ == "__main__":
    cli_main()
