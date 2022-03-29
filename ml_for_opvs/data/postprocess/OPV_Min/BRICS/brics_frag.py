import pkg_resources
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import Draw, rdmolops
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import display
from collections import deque
import numpy as np
import copy
import ast

MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/process/OPV_Min/master_opv_ml_from_min.csv"
)

BRICS_FRAG_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/BRICS/master_brics_frag.csv"
)


class BRIC_FRAGS:
    """
    Class to fragment molecules by BRICS approach using RDKIT
    """

    def __init__(self, datapath):
        """
        Inits BRIC_FRAGS with preprocessed data
        
        Args:
            datapath: path to preprocessed donor-acceptor data
        """
        self.data = pd.read_csv(datapath)

    def tokenize_frag(self, brics_frag):
        """
        Function that tokenizes fragment and returns tokenization to csv file

        Args:
            brics_frag: list of all fragments for donor-acceptor pair molecules
        Returns:
            frag_dict: Dictionary of fragment
            da_pair_tokenized: Pandas series of tokenized fragments
        """
        # create dictionary of fragments
        frag_dict = {"_PAD": 0, ".": 1}
        for index, value in brics_frag.items():
            for frag in value:
                if frag not in frag_dict.keys():
                    frag_dict[frag] = len(frag_dict)

        print(len(frag_dict))  # 190 frags

        # tokenize frags
        da_pair_tokenized = []
        max_seq_length = 1
        for index, value in brics_frag.items():
            tokenized_list = []
            for frag in value:
                # map fragment to value in dictionary
                tokenized_list.append(frag_dict[frag])
            # for padding inputs
            if len(tokenized_list) > max_seq_length:
                max_seq_length = len(tokenized_list)
            da_pair_tokenized.append(tokenized_list)

        # pad tokenized frags
        for da_pair in da_pair_tokenized:
            num_of_pad = max_seq_length - len(da_pair)
            for i in range(num_of_pad):
                da_pair.insert(0, 0)

        return da_pair_tokenized, frag_dict

    def remove_dummy(self, mol):
        """
        Function that removes dummy atoms from mol and returns SMILES

        Args:
            mol: RDKiT mol object for removing dummy atoms (*)
        """
        dummy_idx = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                dummy_idx.append(atom.GetIdx())
        # remove dummy atoms altogether
        ed_mol = Chem.EditableMol(mol)
        ed_mol.BeginBatchEdit()
        for idx in dummy_idx:
            ed_mol.RemoveAtom(idx)
        ed_mol.CommitBatchEdit()
        edited_mol = ed_mol.GetMol()
        return Chem.MolToSmiles(edited_mol)

    def bric_frag(self):
        """
        Fragments molecules (from SMILES) using BRICS from RDKIT

        Args:
            None

        Returns:
            Creates new master_brics_frag.csv with Labels, SMILES, DA_pairs, Fragments, PCE(%)
        """
        brics_df = pd.DataFrame(
            columns=[
                "Donor",
                "Donor_SMILES",
                "Acceptor",
                "Acceptor_SMILES",
                "PCE(%)",
                "Donor_BRICS",
                "Acceptor_BRICS",
                "DA_pair_BRICS",
                "DA_tokenized_BRICS",
                "AD_tokenized_BRICS",
            ]
        )
        brics_df["Donor"] = self.data["Donor"]
        brics_df["Donor_SMILES"] = self.data["Donor_SMILES"]
        brics_df["Acceptor"] = self.data["Acceptor"]
        brics_df["Acceptor_SMILES"] = self.data["Acceptor_SMILES"]
        brics_df["PCE(%)"] = self.data["PCE(%)"]

        # Iterate through row and fragment using BRICS
        # to get Donor_BRICS, Acceptor_BRICS, and DA_pair_BRICS
        for index, row in brics_df.iterrows():
            donor_smi = brics_df.at[index, "Donor_SMILES"]
            acceptor_smi = brics_df.at[index, "Acceptor_SMILES"]
            donor_mol = Chem.MolFromSmiles(donor_smi)
            acceptor_mol = Chem.MolFromSmiles(acceptor_smi)
            donor_brics = list(BRICS.BRICSDecompose(donor_mol, returnMols=True))
            donor_brics_smi = []
            for frag in donor_brics:
                frag_smi = self.remove_dummy(frag)  # remove dummy atoms
                donor_brics_smi.append(frag_smi)
            acceptor_brics_smi = []
            acceptor_brics = list(BRICS.BRICSDecompose(acceptor_mol, returnMols=True))
            for frag in acceptor_brics:
                frag_smi = self.remove_dummy(frag)  # remove dummy atoms
                acceptor_brics_smi.append(frag_smi)

            brics_df.at[index, "Donor_BRICS"] = donor_brics_smi
            print(donor_smi, donor_brics_smi)
            brics_df.at[index, "Acceptor_BRICS"] = acceptor_brics_smi
            # da_pair fragments
            da_pair = donor_brics_smi
            da_pair.append(".")
            da_pair.extend(acceptor_brics_smi)
            brics_df.at[index, "DA_pair_BRICS"] = da_pair

        tokenized_pair_array, frag_dict = self.tokenize_frag(brics_df["DA_pair_BRICS"])
        index = 0
        for da_pair in tokenized_pair_array:
            brics_df.at[index, "DA_tokenized_BRICS"] = da_pair
            index += 1
        brics_df.to_csv(BRICS_FRAG_DATA, index=False)

        return frag_dict

    def frag_visualization(self, frag_dict):
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
            maxMols=200,
            subImgSize=(300, 300),
            legends=frag_legends,
        )
        display(img)


b_frag = BRIC_FRAGS(MASTER_DATA)
frag_dict = b_frag.bric_frag()
# print(frag_dict)
# b_frag.frag_visualization(frag_dict)
