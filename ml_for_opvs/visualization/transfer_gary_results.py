from pathlib import Path
import pandas as pd
import os
import shutil
import re

# os.rename("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
# os.replace("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
# shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")

# gary's trained results folder
gary_trained_results: Path = (
    Path(__file__).parent.parent.parent / "gary" / "trained_results"
)

# training folder with all results
training_folder: Path = Path(__file__).parent.parent / "training" / "OPV_Min"


def transfer_results(gary_trained_results: Path, training_folder: Path):
    """Transfer gary's trained results to the training folder.
    Args:
        gary_trained_results (Path): _description_
        training_folder (Path): _description_
    """
    input_reps: dict = {
        "fp": "fingerprint",
        "brics": "BRICS",
        "homolumo": "homo_lumo",
        "smiles": "smiles",
        "bigsmiles": "smiles",
        "selfies": "smiles",
        "ohe": "ohe",
        "graphembed": "graphembed",
        "mordred": "mordred",
    }
    models: dict = {"gp": "GP", "ngboost": "NGBoost"}
    feature: dict = {
        "brics": "DA_tokenized_BRICS",
        "fp": "DA_FP_radius_3_nbits_1024",
        "gnn": "DA_gnn",
        "graphembed": "DA_graphembed",
        "homolumo": "HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV",
        "mordred": "DA_mordred",
        "ohe": "DA_ohe",
        "smiles": "DA_SMILES",
        "bigsmiles": "DA_BigSMILES",
        "selfies": "DA_SELFIES",
    }
    target: dict = {}
    for file in gary_trained_results.iterdir():
        # move this file to the correct folder in training folder
        absolute_file: str = file.absolute
        name_of_file: str = file.name
        # new path to correct folder, representation / model / feature / target
        new_path: Path = training_folder
        name_of_file_list: list = name_of_file.split(".")[0].split("_")
        # print(f"{name_of_file_list=}")
        try:
            new_path: Path = (
                new_path
                / input_reps[name_of_file_list[2]]
                / "result_molecules_only"
                / models[name_of_file_list[1]]
            )
            new_path: Path = (
                new_path / feature[name_of_file_list[2]] / "calc_PCE_percent"
            )
            # make sure new_path exists
            new_path.mkdir(parents=True, exist_ok=True)
            if "summary" in name_of_file:
                new_path: Path = new_path / "summary.csv"
            else:
                new_path: Path = new_path / "progress_report.csv"
            print(f"{file.absolute=}")
            print(f"{new_path.absolute=}")
            shutil.copyfile(file, new_path)
        except:
            continue
    
def transfer_ensemble_results(gary_trained_results: Path, training_folder: Path):
    """Transfer gary's trained results to the training folder.
    Args:
        gary_trained_results (Path): _description_
        training_folder (Path): _description_
    """
    input_reps: dict = {
        "fp": "fingerprint",
        "brics": "BRICS",
        "homolumo": "homo_lumo",
        "smiles": "smiles",
        "bigsmiles": "smiles",
        "selfies": "smiles",
        "ohe": "ohe",
        "graphembed": "graphembed",
        "mordred": "mordred",
    }
    models: dict = {"gp": "GP_ensemble", "ngboost": "NGBoost_ensemble"}
    feature: dict = {
        "brics": "DA_tokenized_BRICS",
        "fp": "DA_FP_radius_3_nbits_1024",
        "gnn": "DA_gnn",
        "graphembed": "DA_graphembed",
        "homolumo": "HOMO_D_eV,LUMO_D_eV,HOMO_A_eV,LUMO_A_eV",
        "mordred": "DA_mordred",
        "ohe": "DA_ohe",
        "smiles": "DA_SMILES",
        "bigsmiles": "DA_BigSMILES",
        "selfies": "DA_SELFIES",
    }
    target: dict = {}
    for file in gary_trained_results.iterdir():
        # move this file to the correct folder in training folder
        absolute_file: str = file.absolute
        name_of_file: str = file.name
        # new path to correct folder, representation / model / feature / target
        new_path: Path = training_folder
        name_of_file_list: list = name_of_file.split(".")[0].split("_")
        # print(f"{name_of_file_list=}")
        if "ensemble" in name_of_file_list:
            try:
                new_path: Path = (
                    new_path
                    / input_reps[name_of_file_list[2]]
                    / "result_molecules_only"
                    / models[name_of_file_list[1]]
                )
                new_path: Path = (new_path / feature[name_of_file_list[2]])
                if name_of_file_list[3] == "Jsc":
                    new_path: Path = (
                        new_path / "Jsc_mA_cm_pow_neg2"
                    )
                elif name_of_file_list[3] == "Voc":
                    new_path: Path = (new_path / "Voc_V")
                elif name_of_file_list[3] == "FF":
                    new_path: Path = (new_path / "FF_percent")
                elif name_of_file_list[3] == "ensemble":
                    new_path: Path = (new_path / "calc_PCE_percent")             # make sure new_path exists
                new_path.mkdir(parents=True, exist_ok=True)
                if "summary" in name_of_file:
                    new_path: Path = new_path / "summary.csv"
                else:
                    new_path: Path = new_path / "progress_report.csv"
                print(f"{file.absolute=}")
                print(f"{new_path.absolute=}")
                shutil.copyfile(file, new_path)
            except:
                continue


# transfer_results(gary_trained_results, training_folder)
transfer_ensemble_results(gary_trained_results, training_folder)
