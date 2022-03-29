from argparse import ArgumentParser
from pickle import FALSE
import pkg_resources
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error

from opv_ml.ML_models.pytorch.data.OPV_Min.tokenizer import Tokenizer
from opv_ml.ML_models.pytorch.NN.OPV_Min.opv_NN import NNModel
from opv_ml.ML_models.pytorch.LSTM.OPV_Min.opv_LSTM import LSTMModel

from opv_ml.ML_models.pytorch.data.OPV_Min.data import OPVDataModule

import os

from opv_ml.ML_models.pytorch.Transformer.opv_chembert import (
    TransformerModel as TFModel,
)
from opv_ml.ML_models.pytorch.Transformer.opv_chembert_linear import (
    TransformerModel as TFModelLinear,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

DATA_DIR = pkg_resources.resource_filename(
    "opv_ml", "data/process/OPV_Min/master_opv_ml_from_min.csv"
)

FRAG_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/hw_frag/train_frag_master.csv"
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

PREDICTION_PATH = pkg_resources.resource_filename("opv_ml", "data/predictions/",)

MODEL_CHECKPOINT = pkg_resources.resource_filename(
    "opv_ml", "model_checkpoints/OPV_Min"
)

CHEMBERT_TOKENIZER = pkg_resources.resource_filename(
    "opv_ml", "ML_models/pytorch/Transformer/tokenizer_chembert/"
)

CHEMBERT = pkg_resources.resource_filename(
    "opv_ml", "ML_models/pytorch/Transformer/chembert/"
)

SEED_VAL = 4


class Evaluator:
    """Class that contains functions to make predictions/inferences from any model, and visualize results
    """

    def __init__(
        self, model_type: str, model_checkpoint, prediction_csv, model_file, data_dir
    ):
        """
        Parameters
        model_type: type of model such as LR, MLR, NN, RF (backward compatible)
        model_checkpoint: directory of stored checkpoints
        prediction_csv: directory of prediction.csv
        model_file: model checkpoint file path
        """
        self.model_type = model_type
        self.model_checkpoint = model_checkpoint
        self.prediction_csv = prediction_csv
        self.model_file = model_file
        self.data = pd.read_csv(data_dir)

    def prepare_data(self, input_rep) -> None:
        self.data["DA_pair"] = " "
        # concatenate Donor and Acceptor Inputs
        if input_rep == 0:
            representation = "SMILES"
        elif input_rep == 1:
            representation = "Big_SMILES"
        elif input_rep == 2:
            representation = "SELFIES"

        for index, row in self.data.iterrows():
            self.data.at[index, "DA_pair"] = (
                row["Donor_{}".format(representation)]
                + "."
                + row["Acceptor_{}".format(representation)]
            )

    def model_eval(self, input_rep):
        """Function that pre-processes data and loads trained model for
        evaluation and outputs a csv filled with ground truth data and predicted data
        input_rep: 0 - SMILES, 1 - Big_SMILES, 2 - SELFIES
        """
        pl.seed_everything(SEED_VAL)
        # --------------
        # load model
        # --------------
        # concatenate model file path
        final_model = (
            self.model_checkpoint + "\\" + self.model_type + "\\" + self.model_file
        )

        # load trained model depending on model type
        if self.model_type == "NN":
            model = NNModel.load_from_checkpoint(final_model)
        elif self.model_type == "LSTM":
            model = LSTMModel.load_from_checkpoint(final_model)
        elif self.model_type == "Transformer":
            model = TFModel.load_from_checkpoint(
                final_model, pt_model=CHEMBERT, pt_tokenizer=CHEMBERT_TOKENIZER
            )
        elif self.model_type == "Transformer_Linear":
            model = TFModelLinear.load_from_checkpoint(
                final_model, pt_model=CHEMBERT, pt_tokenizer=CHEMBERT_TOKENIZER
            )
        # use model after training or load weights and drop into the production system
        model.eval()

        # ------------
        # data
        # ------------
        parser = ArgumentParser()
        parser.add_argument("--train_batch_size", type=int, default=128)
        parser.add_argument("--val_batch_size", type=int, default=32)
        parser.add_argument("--test_batch_size", type=int, default=32)
        # parser.add_argument("--data_dir", type=str, default=BRICS_MASTER_DATA)
        # parser.add_argument("--data_dir", type=str, default=FRAG_MASTER_DATA)
        parser.add_argument("--data_dir", type=str, default=DATA_DIR)
        # parser.add_argument("--data_dir", type=str, default=MANUAL_MASTER_DATA)
        # parser.add_argument("--data_dir", type=str, default=FP_MASTER_DATA)

        parser.add_argument("--dataloader_num_workers", type=int, default=12)
        args = parser.parse_args()

        # for transformer
        if self.model_type == "Transformer" or self.model_type == "Transformer_Linear":
            chembert_model = CHEMBERT
            chembert_tokenizer = CHEMBERT_TOKENIZER
        else:
            chembert_model = None
            chembert_tokenizer = None
        # ---------------
        # Data Conditions
        # ---------------
        smiles = False
        # smiles = True
        aug_smiles = False
        # aug_smiles = True
        hw_frag = False
        # hw_frag = True
        aug_hw_frag = False
        # aug_hw_frag = True
        # selfies = False
        selfies = True
        brics = False
        # brics = True
        manual = False
        # manual = True
        aug_manual = False
        # aug_manual = True
        fingerprint = False
        # fingerprint = True
        # shuffled = False
        shuffled = True

        fp_radius = 2
        fp_nbits = 512

        data_module = OPVDataModule(
            train_batch_size=args.train_batch_size,
            val_batch_size=args.val_batch_size,
            test_batch_size=args.test_batch_size,
            num_workers=args.dataloader_num_workers,
            smiles=smiles,
            selfies=selfies,
            hw_frag=hw_frag,  # change to aug (in suffix)
            aug_hw_frag=aug_hw_frag,
            aug_smiles=aug_smiles,  # change to aug_smi (in suffix)
            brics=brics,
            manual=manual,
            aug_manual=aug_manual,
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

        # create new dataframe with predicted and ground truth results
        test_df = pd.DataFrame()
        # add empty columns
        test_df["Experimental_PCE_(%)"] = np.nan
        test_df["Predicted_PCE_(%)"] = np.nan
        test_set = data_module.pce_test
        print(len(test_set))
        row_index = 0
        for i in test_set.indices:
            experimental_pce = test_set.dataset.y[i]
            experimental_input = test_set.dataset.x[i]
            experimental_input = np.expand_dims(experimental_input, 0)
            experimental_input = torch.from_numpy(experimental_input)
            predicted_pce = model(experimental_input)
            test_df.at[row_index, "Experimental_PCE_(%)"] = experimental_pce
            test_df.at[row_index, "Predicted_PCE_(%)"] = predicted_pce
            row_index += 1

        # prepare file path for storing the predictions
        prediction_file = self.prediction_csv + self.model_type + "_predictions.csv"
        test_df.to_csv(prediction_file, float_format="%.3f")
        self.new_predictions = test_df

    def parity_plot(self):
        """Creates a parity plot for two column data (predicted data and ground truth data)"""
        # find slope and y-int of linear line of best fit
        m, b = np.polyfit(
            self.new_predictions["Experimental_PCE_(%)"],
            self.new_predictions["Predicted_PCE_(%)"],
            1,
        )
        print(m, b)
        # find correlation coefficient
        corr_coef = np.corrcoef(
            self.new_predictions["Experimental_PCE_(%)"],
            self.new_predictions["Predicted_PCE_(%)"],
        )[0, 1]
        # find rmse
        rmse = np.sqrt(
            mean_squared_error(
                self.new_predictions["Experimental_PCE_(%)"],
                self.new_predictions["Predicted_PCE_(%)"],
            )
        )

        fig, ax = plt.subplots()
        ax.set_title("Predicted vs. Experimental PCE (%)")
        ax.set_xlabel("Experimental_PCE_(%)")
        ax.set_ylabel("Predicted_PCE_(%)")
        ax.scatter(
            self.new_predictions["Experimental_PCE_(%)"],
            self.new_predictions["Predicted_PCE_(%)"],
            s=4,
            alpha=0.7,
            color="#0AB68B",
        )
        ax.plot(
            self.new_predictions["Experimental_PCE_(%)"],
            m * self.new_predictions["Experimental_PCE_(%)"] + b,
            color="black",
            label=["R: " + str(corr_coef) + "_" + "RMSE: " + str(rmse)],
        )
        ax.plot([0, 1], [0, 1], "--", color="blue", label="Perfect Correlation")
        ax.legend(loc="upper left")
        # add text box with slope and y-int
        # textstr = "slope: " + str(m) + "\n" + "y-int: " + str(b)
        # ax.text(0.5, 0.5, textstr, wrap=True, verticalalignment="top")
        plt.show()

    def MSE_predictions(self):
        y_true = self.new_predictions["Experimental_PCE_(%)"]
        y_pred = self.new_predictions["Predicted_PCE_(%)"]

        corr_coef = np.corrcoef(
            self.new_predictions["Experimental_PCE_(%)"],
            self.new_predictions["Predicted_PCE_(%)"],
        )[0, 1]
        # MSE is heavily affected by large outliers/differences at stepscore>600 range
        mse = np.square(np.subtract(y_true, y_pred)).mean()
        rmse = np.sqrt(mse)
        print("R: ", corr_coef, " RMSE:", rmse)


def cli_main():
    # model = "LSTM"
    # model = "Transformer"
    model = "NN"
    # linear = True
    linear = False
    if model == "LSTM":
        working_eval = Evaluator(
            "LSTM",
            MODEL_CHECKPOINT,
            PREDICTION_PATH,
            "smi_shuffled_LSTM-epoch=624-val_loss=0.052-n_hidden=256-n_embedding=256-drop_prob=0.3-lr=0.01-train_batch_size=256.ckpt",
            DATA_DIR,
        )
    elif model == "Transformer":
        if linear:
            working_eval = Evaluator(
                "Transformer_Linear",
                MODEL_CHECKPOINT,
                PREDICTION_PATH,
                "aug_smi_ChemBERTL-epoch=12-val_loss_mse=0.036-drop_prob=0.3-lr=0.01-train_batch_size=128.ckpt",
                DATA_DIR,
            )
        else:
            working_eval = Evaluator(
                "Transformer",
                MODEL_CHECKPOINT,
                PREDICTION_PATH,
                "selfies_shuffled_NN-epoch=218-val_loss=0.055-n_hidden=256-n_embedding=256-drop_prob=0.3-lr=0.001-train_batch_size=128.ckpt",
                DATA_DIR,
            )
    else:
        working_eval = Evaluator(
            "NN",
            MODEL_CHECKPOINT,
            PREDICTION_PATH,
            "selfies_shuffled_NN-epoch=218-val_loss=0.055-n_hidden=256-n_embedding=256-drop_prob=0.3-lr=0.001-train_batch_size=128.ckpt",
            DATA_DIR,
        )
    working_eval.model_eval(2)  # 0 - SMILES, 2 - SELFIES
    working_eval.MSE_predictions()
    working_eval.parity_plot()


if __name__ == "__main__":
    cli_main()
