from doctest import master
from rdkit import Chem
import rdkit
from rdkit.Chem import Draw, rdchem
import pkg_resources
import pandas as pd
import ast
import copy
from collections import deque
from IPython.display import display
import numpy as np

DONOR_DIR = pkg_resources.resource_filename(
    "ml_for_opvs", "data/preprocess/OPV_Min/clean_min_donors.csv"
)

ACCEPTOR_DIR = pkg_resources.resource_filename(
    "ml_for_opvs", "data/preprocess/OPV_Min/clean_min_acceptors.csv"
)

IMG_PATH = pkg_resources.resource_filename(
    "ml_for_opvs", "data/postprocess/OPV_Min/manual_frag/"
)

FRAG_DONOR_DIR = pkg_resources.resource_filename(
    "ml_for_opvs", "data/postprocess/OPV_Min/manual_frag/donor_frags.csv"
)

FRAG_ACCEPTOR_DIR = pkg_resources.resource_filename(
    "ml_for_opvs", "data/postprocess/OPV_Min/manual_frag/acceptor_frags.csv"
)

# For Manual Fragments!
MASTER_MANUAL_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/postprocess/OPV_Min/manual_frag/master_manual_frag.csv"
)

OPV_DATA = pkg_resources.resource_filename(
    "ml_for_opvs",
    "data/process/OPV_Min/Machine Learning OPV Parameters - device_params.csv",
)

MASTER_ML_DATA = pkg_resources.resource_filename(
    "ml_for_opvs", "data/postprocess/OPV_Min/master_ml_for_opvs_from_min.csv",
)


class manual_frag:
    "Class that contains functions necessary to fragment molecules any way you want"

    def __init__(self, opv_data, donor_data, acceptor_data):
        """
        Instantiate class with appropriate data.

        Args:
            opv_data: path to ML data downloaded from Google Drive shared w/ UCSB
            donor_data: path to preprocessed donor data
            acceptor_data: path to preprocessed acceptor data

        Returns:
            None
        """
        self.donor_data = pd.read_csv(donor_data)
        self.acceptor_data = pd.read_csv(acceptor_data)
        self.opv_data = pd.read_csv(opv_data)

    # pipeline
    # 1 iterate with index (main)
    # 2 show molecule with atom.index
    # 3 ask for begin/end atom index OR bond index
    # 4 fragment
    # 5 show fragmented molecule
    # 6 if correct, convert to smiles and store in new .csv
    # 7 if incorrect, go back to step 3
    # 8 NOTE: be able to manually look up any donor/acceptor and re-fragment

    def lookup(self, group_type: str, index: int) -> str:
        """
        Function that finds and returns SMILES from donor or acceptor .csv
        
        Args:
            group_type: choose between donor and acceptor
            index: index of row in dataframe

        Returns:
            smi: SMILES of looked up molecule
        """
        if group_type == "donor":
            try:
                smi = self.donor_data.at[index, "SMILES"]
            except:
                print(
                    "Max index exceeded, please try again. Max index is: ",
                    len(self.donor_data["SMILES"]) - 1,
                )
        elif group_type == "acceptor":
            try:
                smi = self.acceptor_data.at[index, "SMILES"]
            except:
                print(
                    "Max index exceeded, please try again. Max index is: ",
                    len(self.acceptor_data["SMILES"]) - 1,
                )

        return smi

    def fragmenter(smi: str, mol_type: str):
        """
        Function that asks user how to fragment molecule

        Args:
            smi: SMILES to fragment
            mol_type: fragmenting donor or acceptor?
        
        Returns:
            ordered_frag: molecule that was fragmented by user's input, and properly ordered
        """
        mol = Chem.MolFromSmiles(smi)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
        drawn = Draw.MolToFile(mol, IMG_PATH + "manual.png", size=(700, 700))
        fragmented = False
        reformed = False
        # show all bond indexes with corresponding begin/atom idx
        for bond in mol.GetBonds():
            print(
                "bond: ",
                bond.GetIdx(),
                "begin, end: ",
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
            )

        while not fragmented:
            # Ex. 30, 31, 33, 34, 101, 102
            frag_idx = input("Begin/End Atom Indexes of bond to be fragmented: ")
            if frag_idx == "None":
                mol_frag = mol
                break
            frag_tuple = tuple(map(int, frag_idx.split(", ")))
            mol_frag = Chem.FragmentOnBonds(mol, frag_tuple, addDummies=False)
            for atom in mol_frag.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx())
            drawn = Draw.MolToFile(
                mol_frag, IMG_PATH + "manual_frag.png", size=(700, 700)
            )
            if mol_type == "acceptor":
                # add new bonds to re-form the ring structure
                rwmol_frag = Chem.RWMol(mol_frag)
                while not reformed:
                    # add extra atoms to help re-form rings
                    num_add_atom = int(input("Number of atoms to be added: "))
                    rwmol_frag.BeginBatchEdit()
                    for i in range(num_add_atom):
                        atom = Chem.MolFromSmiles("C").GetAtomWithIdx(0)
                        atom_idx = rwmol_frag.AddAtom(atom)
                        print("new_atom_idx: ", atom_idx)
                    rwmol_frag.CommitBatchEdit()
                    mol_frag = rwmol_frag.GetMol()
                    for atom in mol_frag.GetAtoms():
                        atom.SetAtomMapNum(atom.GetIdx())
                    drawn = Draw.MolToFile(
                        mol_frag, IMG_PATH + "manual_frag.png", size=(700, 700)
                    )
                    if num_add_atom != 0:
                        add_bond = input(
                            "Begin/End Atom Indexes of bond to be added?: "
                        )
                        add_bond_list = add_bond.split(", ")
                        rwmol_frag.BeginBatchEdit()
                        bond_input = int(input("Bond Type?: "))
                        if bond_input == 0:  # SINGLE
                            bond_mol = Chem.MolFromSmiles("CC")
                            bond = bond_mol.GetBonds()[0]
                            bond_type = bond.GetBondType()
                        elif bond_input == 1:  # AROMATIC
                            bond_mol = Chem.MolFromSmiles("c1ccccc1")
                            bond = bond_mol.GetBonds()[0]
                            bond_type = bond.GetBondType()
                        elif bond_input == 2:  # DOUBLE
                            bond_mol = Chem.MolFromSmiles("C=C")
                            bond = bond_mol.GetBonds()[0]
                            bond_type = bond.GetBondType()
                        for i in range(0, len(add_bond_list), 2):
                            rwmol_frag.AddBond(
                                int(add_bond_list[i]),
                                int(add_bond_list[i + 1]),
                                order=bond_type,
                            )
                        rwmol_frag.CommitBatchEdit()
                    mol_frag = rwmol_frag.GetMol()
                    drawn = Draw.MolToFile(
                        mol_frag, IMG_PATH + "manual_frag.png", size=(700, 700)
                    )
                    reformed = input("Are all the rings reformed correctly?: ")
                    if reformed == "y":
                        reformed = True
            correct = input("Is the molecule fragmented correctly?: ")
            if correct == "y":
                fragmented = True

        # removes atom map numbering
        [a.SetAtomMapNum(0) for a in mol_frag.GetAtoms()]
        # replace dummy atoms
        edmol_frag = Chem.EditableMol(mol_frag)
        c_atom = Chem.MolFromSmiles("C").GetAtomWithIdx(0)
        edmol_frag.BeginBatchEdit()
        [
            edmol_frag.ReplaceAtom(atom.GetIdx(), c_atom)
            for atom in mol_frag.GetAtoms()
            if atom.GetAtomicNum() == 0
        ]
        edmol_frag.CommitBatchEdit()
        final_mol = edmol_frag.GetMol()
        drawn = Draw.MolToFile(final_mol, IMG_PATH + "manual_frag.png", size=(700, 700))
        frag_smi = Chem.MolToSmiles(final_mol)
        frag_list = frag_smi.split(".")

        # order the fragments
        frag_length = len(frag_list)
        # put placeholders
        ordered = False
        while not ordered:
            ordered_frag = []
            for i in range(frag_length):
                ordered_frag.append(i)
            for frag in frag_list:
                order_idx = int(input("Ordering of current frag (" + str(frag) + "):"))
                ordered_frag[order_idx] = frag
            print(ordered_frag)
            correct = input("Are the fragments ordered correctly?: ")
            if correct == "y":
                ordered = True

        return ordered_frag

    def fragment_new(self, frag_path, master_path, mol_type):
        """
        Function that finds index of missing donors/acceptors in the fragmented file.

        Args:
            frag_path: path to file with pre-existing fragmented data
            master_path: path to file with all data (OPV ML data)
            mol_type: option for donors and acceptors
        
        Returns:
            .csv file with new molecules that was fragmented by user's input, and properly ordered
        """
        frag_df = pd.read_csv(frag_path)
        master_df = pd.read_csv(master_path)

        if mol_type == "donor":
            for index, row in master_df.iterrows():
                if master_df.at[index, "Donor"] not in list(frag_df["Label"]):
                    smi = master_df.at[index, "Donor_SMILES"]
                    frag_list = self.fragmenter(smi, mol_type)
                    temp_df = pd.DataFrame(
                        {
                            "Label": master_df.at[index, "Donor"],
                            "SMILES": smi,
                            "Fragments": [frag_list],
                        }
                    )
                    frag_df = pd.concat([frag_df, temp_df], ignore_index=True, axis=0)
                    frag_df.to_csv(frag_path, index=False)
        elif mol_type == "acceptor":
            for index, row in master_df.iterrows():
                if master_df.at[index, "Acceptor"] not in list(frag_df["Label"]):
                    print(master_df.at[index, "Acceptor"])
                    smi = master_df.at[index, "Acceptor_SMILES"]
                    frag_list = self.fragmenter(smi, mol_type)
                    temp_df = pd.DataFrame(
                        {
                            "Label": master_df.at[index, "Acceptor"],
                            "SMILES": smi,
                            "Fragments": [frag_list],
                        }
                    )
                    frag_df = pd.concat([frag_df, temp_df], ignore_index=True, axis=0)
                    frag_df.to_csv(frag_path, index=False)

    def new_frag_files(self, donor_frag_dir, acceptor_frag_dir):
        """
        Creates empty .csv files for donor frags and acceptor frags
        """
        donor_frag = pd.DataFrame(columns=["Label", "SMILES", "Fragments"])
        acceptor_frag = pd.DataFrame(columns=["Label", "SMILES", "Fragments"])

        donor_frag["Label"] = self.donor_data["Label"]
        donor_frag["SMILES"] = self.donor_data["SMILES"]
        donor_frag["Fragments"] = " "

        acceptor_frag["Label"] = self.acceptor_data["Label"]
        acceptor_frag["SMILES"] = self.acceptor_data["SMILES"]
        acceptor_frag["Fragments"] = " "

        donor_frag.to_csv(donor_frag_dir, index=False)
        acceptor_frag.to_csv(acceptor_frag_dir, index=False)

def return_frag_dict(donor_frag_dir, acceptor_frag_dir):
    """
    Sifts through manual fragments and creates unique dictionary of frag2idx

    Args:
        None
    
    Returns:
        frag_dict: dictionary of unique fragments in the combination of donor and acceptor fragmented molecules
    """
    donor_frag: pd.DataFrame = pd.read_csv(donor_frag_dir)
    acceptor_frag: pd.DataFrame = pd.read_csv(acceptor_frag_dir)

    frag_dict = {}
    frag_dict["_PAD"] = 0
    frag_dict["."] = 1
    id = len(frag_dict)
    for i in range(len(donor_frag)):
        frag_str = ast.literal_eval(donor_frag.at[i, "Fragments"])
        frag_list = frag_str
        for frag in frag_list:
            if frag not in list(frag_dict.keys()):
                frag_dict[frag] = id
                id += 1

    for i in range(len(acceptor_frag)):
        frag_str = ast.literal_eval(acceptor_frag.at[i, "Fragments"])
        frag_list = frag_str
        for frag in frag_list:
            if frag not in list(frag_dict.keys()):
                frag_dict[frag] = id
                id += 1

    return frag_dict

def tokenize_frag(list_of_frag, frag_dict, max_seq_length):
    """
    Tokenizes input list of fragment from given dictionary
    * Assumes frag_dict explains all of list_of_frig

    Args:
        list_of_frag: list of all the fragments for tokenization
        frag_dict: dictionary of unique fragments from donor and acceptor molecules
        max_seq_length: the largest number of fragments for one molecule
    """
    tokenized_list = []
    # Add pre-padding
    num_of_pad = max_seq_length - len(list_of_frag)
    for i in range(num_of_pad):
        tokenized_list.append(0)

    for frag in list_of_frag:
        tokenized_list.append(frag_dict[frag])

    return tokenized_list

def create_manual_json(donor_data_path, acceptor_data_path, opv_data_path, frag_dict, master_manual_path):
    """
    Creates master data file for manual frags

    Args:
        frag_dict: dictionary of unique fragments from donor and acceptor molecules
        master_manual_path: path to master .csv file for training on manual fragments
    """
    donor_data = pd.read_csv(donor_data_path)
    acceptor_data = pd.read_csv(acceptor_data_path)
    opv_data = pd.read_csv(opv_data_path)
    manual_df = pd.read_csv(opv_data_path)
    manual_df["DA_manual_tokenized"] = ""
    manual_df["DA_manual_tokenized_aug"] = ""
    

    donor_avail = list(donor_data["Label"])
    acceptor_avail = list(acceptor_data["Label"])
    # iterate through data_from_min.csv for donor-acceptor pairs
    for index, row in opv_data.iterrows():
        # only keep the rows with available donor and acceptor molecules from clean donors and acceptors
        if (row["Donor"] in donor_avail) and (row["Acceptor"] in acceptor_avail):
            # get SMILES of donor and acceptor
            donor_row = donor_data.loc[
                donor_data["Label"] == row["Donor"]
            ]
            donor_smile = donor_row["SMILES"].values[0]
            donor_big_smi = donor_row["Donor_BigSMILES"].values[0]
            acceptor_row = acceptor_data.loc[
                acceptor_data["Label"] == row["Acceptor"]
            ]
            acceptor_smile = acceptor_row["SMILES"].values[0]
            acceptor_big_smi = acceptor_row["Acceptor_BigSMILES"].values[0]


    # find max_seq_length
    max_seq_length = 0
    for i in range(len(manual_df)):
        donor_label = manual_df.at[i, "Donor"]
        acceptor_label = manual_df.at[i, "Acceptor"]
        donor_row = donor_data.loc[donor_data["Label"] == donor_label]
        acceptor_row = acceptor_data.loc[
            acceptor_data["Label"] == acceptor_label
        ]
        donor_frags = ast.literal_eval(donor_row["Fragments"].values[0])
        acceptor_frags = ast.literal_eval(acceptor_row["Fragments"].values[0])
        max_frag_list = donor_frags
        max_frag_list.append(".")
        max_frag_list.extend(acceptor_frags)
        max_frag_length = len(max_frag_list)
        if max_frag_length > max_seq_length:
            max_seq_length = max_frag_length

    print("max_frag_length: ", max_seq_length)

    for i in range(len(manual_df)):
        donor_label = manual_df.at[i, "Donor"]
        acceptor_label = manual_df.at[i, "Acceptor"]
        donor_row = donor_data.loc[donor_data["Label"] == donor_label]
        acceptor_row = acceptor_data.loc[
            acceptor_data["Label"] == acceptor_label
        ]
        donor_frags = ast.literal_eval(donor_row["Fragments"].values[0])
        acceptor_frags = ast.literal_eval(acceptor_row["Fragments"].values[0])

        # DA Pairs
        da_pair_frags = copy.copy(donor_frags)
        da_pair_frags.append(".")
        da_pair_frags.extend(acceptor_frags)
        da_pair_tokenized = tokenize_frag(
            da_pair_frags, frag_dict, max_seq_length
        )

        # AUGMENT Donor (pre-ordered)
        augmented_donor_list = []
        donor_frag_deque = deque(copy.copy(donor_frags))
        for j in range(len(donor_frags)):
            frag_rotate = copy.copy(donor_frag_deque)
            frag_rotate.rotate(j)
            frag_rotate = list(frag_rotate)
            augmented_donor_list.append(frag_rotate)

        # DA Pairs augmented
        da_pair_tokenized_aug = []
        for aug_donors in augmented_donor_list:
            da_aug_pair = copy.copy(aug_donors)
            da_aug_pair.append(".")
            da_aug_pair.extend(acceptor_frags)
            da_aug_tokenized = tokenize_frag(
                da_aug_pair, frag_dict, max_seq_length
            )
            da_pair_tokenized_aug.append(da_aug_tokenized)

        # ADD TO MANUAL DF
        # print(type(da_pair_tokenized))
        manual_df["DA_manual_tokenized"] = manual_df["DA_manual_tokenized"].astype(
            "object"
        )
        manual_df["DA_manual_tokenized_aug"] = manual_df[
            "DA_manual_tokenized_aug"
        ].astype("object")
        manual_df.at[i, "DA_manual_tokenized"] = da_pair_tokenized
        manual_df.at[i, "DA_manual_tokenized_aug"] = da_pair_tokenized_aug

    manual_df.to_csv(master_manual_path, index=False)

def bigsmiles_from_frag(donor_frag_path, acceptor_frag_path):
    """
    Function that takes ordered fragments (manually by hand) and converts it into BigSMILES representation, specifically block copolymers
    Args:
        donor_frag_path: path to data with manually fragmented donors
        acceptor_frag_path: path to data with manually fragmented acceptors

    Returns:
        concatenates manual fragments into BigSMILES representation and returns to donor/acceptor data
    """
    # donor BigSMILES
    donor_df = pd.read_csv(donor_frag_path)
    donor_df["Donor_BigSMILES"] = ""
    for index, row in donor_df.iterrows():
        donor_big_smi = ""
        position = 0
        for frag in ast.literal_eval(donor_df["Fragments"][index]):
            if position == 0:
                donor_big_smi += "{[][<]"
                donor_big_smi += str(frag)
            elif (
                position == len(donor_df["Fragments"][index]) - 1
            ):
                donor_big_smi += str(frag)
                donor_big_smi += "[>][]}"
            else:
                donor_big_smi += "[>][<]}{[>][<]"
                donor_big_smi += str(frag)
            position += 1
        donor_df["Donor_BigSMILES"][index] = donor_big_smi

    donor_df.to_csv(donor_frag_path, index=False)

    # acceptor BigSMILES
    acceptor_df = pd.read_csv(acceptor_frag_path)
    acceptor_df["Acceptor_BigSMILES"] = ""
    # NOTE: NFAs are not polymers, therefore BigSMILES = SMILES
    acceptor_df["Acceptor_BigSMILES"] = acceptor_df["SMILES"]

    acceptor_df.to_csv(acceptor_frag_path, index=False)

def frag_visualization(frag_dict):
    """
    Visualizes the dictionary of unique fragments
    NOTE: use in jupyter notebook

    Args:
        dictionary of unique fragments from donor and acceptor molecules
    
    Returns:
        img: image of all the unique fragments
    """
    print(len(frag_dict))
    frag_list = [Chem.MolFromSmiles(frag) for frag in frag_dict.keys()]
    frag_legends = []
    for frag_key in frag_dict.keys():
        label = str(frag_dict[frag_key])
        frag_legends.append(label)

    img = Draw.MolsToGridImage(
        frag_list,
        molsPerRow=20,
        maxMols=400,
        subImgSize=(300, 300),
        legends=frag_legends,
    )
    display(img)


def fragment_files():
    manual = manual_frag(MASTER_ML_DATA, DONOR_DIR, ACCEPTOR_DIR)

    # # NOTE: DO NOT USE IF FRAGMENTED
    # manual.new_frag_files(
    #     FRAG_DONOR_DIR, FRAG_ACCEPTOR_DIR
    # )  # do it only the first time

    # iterate through donor and acceptor files
    # donor_df = pd.read_csv(FRAG_DONOR_DIR)
    # for i in range(0, len(donor_df["SMILES"])):  # len(donor_df["SMILES"])
    #     smi = manual.lookup("donor", i)
    #     frag_list = manual.fragmenter(smi, "donor")
    #     donor_df.at[i, "Fragments"] = frag_list
    #     donor_df.to_csv(FRAG_DONOR_DIR, index=False)

    # acceptor_df = pd.read_csv(FRAG_ACCEPTOR_DIR)
    # for i in range(0, len(acceptor_df["SMILES"])):
    #     smi = manual.lookup("acceptor", i)
    #     frag_list = manual.fragmenter(smi, "acceptor")
    #     acceptor_df.at[i, "Fragments"] = frag_list
    #     acceptor_df.to_csv(FRAG_ACCEPTOR_DIR, index=False)

    # fragment by unique/new donors/acceptors
    # prints index of donors that have not been fragmented.
    # For Donors
    manual.fragment_new(FRAG_DONOR_DIR, MASTER_ML_DATA, "donor")
    # For Acceptors
    manual.fragment_new(FRAG_ACCEPTOR_DIR, MASTER_ML_DATA, "acceptor")

def export_manual_frag():
    # prepare manual frag data
    frag_dict = return_frag_dict(FRAG_DONOR_DIR, FRAG_ACCEPTOR_DIR)
    print(frag_dict)
    # # print(len(frag_dict))
    bigsmiles_from_frag(FRAG_DONOR_DIR, FRAG_ACCEPTOR_DIR)
    create_manual_json(FRAG_DONOR_DIR, FRAG_ACCEPTOR_DIR, MASTER_ML_DATA, frag_dict, MASTER_MANUAL_DATA)


if __name__ == "__main__":
    # fragment_files()
    # export_manual_frag()
    pass
