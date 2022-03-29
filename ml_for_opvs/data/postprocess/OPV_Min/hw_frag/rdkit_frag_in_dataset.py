import pkg_resources
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, rdmolops
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import display
from collections import deque
import numpy as np
import copy
import ast

# NOTE: rdkit_frag_pairs.py for fragmenting molecules to train model

# IPythonConsole.ipython_useSVG = True
pd.set_option("display.max_colwidth", None)

MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/process/OPV_Min/master_opv_ml_from_min.csv"
)

IMAGE_PATH = pkg_resources.resource_filename("opv_ml", "data/postprocess/OPV_Min/")

FRAG_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/frag_master_opv_ml_from_min.csv"
)

NUM_FRAG_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/num_of_frag.csv"
)

TRAIN_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/OPV_Min/train_frag_master.csv"
)


class Fragger:
    """Class that contains functions to fragment donor-acceptor molecules"""

    def __init__(self, data):
        """
        Inits Fragger with preprocessed data
        
        Args:
            data: path to preprocessed donor-acceptor data
        """
        self.data = pd.read_csv(data)

    def augment_donor_polymer(self, donor_frag_list, bond_idx_list):
        """
        Function that returns list of data augmented donor fragments in SMILES
      
        Args:
            donor_frag_list: list of unordered donor fragments created from donor molecule (1-4 fragments)
            bond_idx_list: list of bond indices
      
        Returns:
            aug_donor_list: list of donor SMILES with different arrangment of fragments
        """
        # sort order by how original molecule is combined
        # NOTE: can combine on any side because rotatable bond (double-check)
        order_index = 0
        ordered_frag_dict = {}
        # find first fragment in original molecule
        for frag_mol in donor_frag_list:
            num_of_bonds = 0
            other_bond_idx_list = []
            for atom in frag_mol.GetAtoms():
                if len(list(atom.GetPropNames())) != 0:
                    bond_idx = atom.GetUnsignedProp("og_idx")
                    result_list_tuples = [
                        idx for idx in bond_idx_list if bond_idx in idx
                    ]
                    result = [item for tuple in result_list_tuples for item in tuple]
                    if len(result) != 0:
                        result.remove(bond_idx)
                        other_bond_idx_list.extend(result)
            for frag_mol_2 in donor_frag_list:
                for atom_2 in frag_mol_2.GetAtoms():
                    if len(list(atom_2.GetPropNames())) != 0:
                        bond_idx_2 = atom_2.GetUnsignedProp("og_idx")
                        if bond_idx_2 in other_bond_idx_list:
                            num_of_bonds += 1
            if num_of_bonds == 1:
                donor_frag_list.remove(frag_mol)
                first_mol_frag = frag_mol
                ordered_frag_dict[order_index] = Chem.MolToSmiles(frag_mol)
                order_index += 1
                break

        # start from first fragment and start building order
        loop_index = 0
        while len(donor_frag_list) != 0:
            # loop through donor_frag_list until there are no more fragments
            try:
                loop_mol = donor_frag_list[loop_index]
            except:
                loop_index = 0
                continue
            correct_frag = False
            for atom in loop_mol.GetAtoms():
                if len(list(atom.GetPropNames())) != 0:
                    bond_idx = atom.GetUnsignedProp("og_idx")
                    if bond_idx in other_bond_idx_list:
                        ordered_frag_dict[order_index] = Chem.MolToSmiles(
                            loop_mol
                        )  # add frag to ordered dictionary
                        order_index += 1
                        donor_frag_list.remove(
                            loop_mol
                        )  # remove from original donor fragment list
                        old_bond_idx = bond_idx
                        correct_frag = True
            if correct_frag:
                for atom in loop_mol.GetAtoms():
                    if len(list(atom.GetPropNames())) != 0:
                        bond_idx_2 = atom.GetUnsignedProp("og_idx")
                        # get bond_idx to next fragment
                        if bond_idx_2 != old_bond_idx:
                            result_list_tuples = [
                                idx for idx in bond_idx_list if bond_idx_2 in idx
                            ]
                            result = [
                                item for tuple in result_list_tuples for item in tuple
                            ]
                            if len(result) != 0:
                                result.remove(bond_idx_2)  # update other bond idx
                                other_bond_idx_list = result
            loop_index += 1

        # create all combinations of fragments while maintaing proper order
        aug_donor_list = []
        ordered_frag_dict_keys = list(ordered_frag_dict.keys())
        frag_dict_deque = deque(ordered_frag_dict_keys)
        for i in range(len(ordered_frag_dict_keys)):
            frag_dict_deque_rotate = copy.copy(frag_dict_deque)
            frag_dict_deque_rotate.rotate(i)
            rotated_frag_dict_list = list(frag_dict_deque_rotate)
            rotated_donor_list = []
            for key in rotated_frag_dict_list:
                rotated_donor_list.append(ordered_frag_dict[key])
            aug_donor_list.append(rotated_donor_list)
        return aug_donor_list

    def donor_frag(self, frag_data, num_of_frag_data, data_aug_bool):
        """
        Fragmenting molecules based on the donor molecules.
        Option: Augment data by keeping fragment order but starting at different fragments

        Args:
            frag_data: path to final .csv file for fragmented data
            num_of_frag_data: distribution of fragments per molecule
            data_aug_bool: boolean for augmenting donor polymers
        
        Returns:
            frag_master_opv_ml.csv: New column with list of fragments, sorted into categories, and (optionally) augmented.
            Dictionary containing all unique fragments with number of occurrences in the dataset.
        """
        donor_dict = {}
        donor_smiles_data = self.data["Donor_SMILES"]
        donor_label_data = self.data["Donor"]
        self.data.insert(7, "Donor_Fragments", " ")
        self.data.insert(8, "Donor_Backbone_Fragments", " ")
        self.data.insert(9, "Donor_Core_Fragments", " ")
        self.data.insert(10, "Donor_Edge_Fragments", " ")
        self.data.insert(11, "Donor_SMILES_augmented", " ")

        # STATS: documenting distribution of number of lengths
        donor_len_df = pd.DataFrame(columns=["Donor_Label", "Donor_Num_Frags"])

        # iterate through all donor SMILES
        index = 0
        bad_label = []
        # swap between donor_smiles_data (for R grp) and donor_smiles_frag_data (for no R grp)
        for donor in donor_smiles_data:
            donor_mol = Chem.MolFromSmiles(donor)
            # add indices to molecule
            # for atom in donor_mol.GetAtoms():
            #     atom.SetAtomMapNum(atom.GetIdx())
            # fragment molecule by bond indices
            # fragment conditions:
            # atom degree = 3 (bond-type independent)
            # atom atomic number = 6 (Carbon)
            # bond type = SINGLE, not in ring
            bond_idx_list = []
            all_dummy_idx = []
            for bond in donor_mol.GetBonds():
                if str(bond.GetBondType()) == "SINGLE" and bond.IsInRing() == False:
                    idx0 = bond.GetBeginAtomIdx()
                    idx1 = bond.GetEndAtomIdx()
                    atom0 = donor_mol.GetAtomWithIdx(idx0)
                    atom1 = donor_mol.GetAtomWithIdx(idx1)
                    if (
                        atom0.GetDegree() == 3
                        and atom0.GetAtomicNum() == 6
                        and atom1.GetDegree() == 3
                        and atom1.GetAtomicNum() == 6
                        and atom0.IsInRing()
                        and atom1.IsInRing()
                    ):
                        bond_idx = bond.GetIdx()
                        bond_idx_list.append(bond_idx)
                        all_dummy_idx.append((idx0, idx1))  # atom index of broken bonds
                        # set original index property for atoms from broken bonds
                        atom0.SetUnsignedProp("og_idx", idx0)
                        atom1.SetUnsignedProp("og_idx", idx1)
                    elif (
                        (atom0.GetDegree() == 1 or atom1.GetDegree() == 1)
                        and (atom0.GetIsAromatic() or atom1.GetIsAromatic())
                        and (atom0.GetAtomicNum() == 6 and atom1.GetAtomicNum() == 6)
                    ):
                        bond_idx = bond.GetIdx()
                        bond_idx_list.append(bond_idx)
                        all_dummy_idx.append((idx0, idx1))  # atom index of broken bonds
                        # set original index property for atoms from broken bonds
                        atom0.SetUnsignedProp("og_idx", idx0)
                        atom1.SetUnsignedProp("og_idx", idx1)

            # check if SetUnsignedProp worked
            # for atoms in donor_mol.GetAtoms():
            #     if len(list(atoms.GetPropNames())) != 0:
            #         print(atoms.GetUnsignedProp("og_idx"))

            frag_mol = rdmolops.FragmentOnBonds(
                donor_mol, bond_idx_list, addDummies=True
            )

            func_grp = []
            backbone_grp = []
            methyl_grp = []
            list_frag_mols = list(rdmolops.GetMolFrags(frag_mol, asMols=True))
            methyl_substruct = Chem.MolFromSmiles("C")
            for mol in list_frag_mols:
                # edge group rule
                # remove dummy atom to check for substructure
                current_frag_edmol = Chem.EditableMol(mol)
                remove_idx = []
                for atom in current_frag_edmol.GetMol().GetAtoms():
                    if atom.GetAtomicNum() == 0:
                        atom_idx = atom.GetIdx()
                        remove_idx.append(atom_idx)

                # performs action all at once so index doesn't change
                current_frag_edmol.BeginBatchEdit()
                for idx in remove_idx:
                    current_frag_edmol.RemoveAtom(idx)
                current_frag_edmol.CommitBatchEdit()

                if methyl_substruct.HasSubstructMatch(current_frag_edmol.GetMol()):
                    methyl_grp.append(mol)
                else:
                    n_dummy = 0
                    # core/functional group rules
                    for atom in mol.GetAtoms():
                        if len(list(atom.GetPropNames())) != 0:
                            if list(atom.GetPropNames())[0] != "dummyLabel":
                                n_dummy += 1
                    if n_dummy == 1:  # functional group
                        func_grp.append(mol)
                    elif n_dummy > 1:  # backbone group
                        backbone_grp.append(mol)

            backbone_fragments = []
            core_fragments = []
            edge_fragments = []

            if data_aug_bool:
                frag_augment = []
                all_augment_idx = copy.copy(all_dummy_idx)

            dummy = True
            for mol in backbone_grp:
                # get original index of dummy atom
                func_bool = False
                for atom in mol.GetAtoms():
                    if len(list(atom.GetPropNames())) != 0:
                        if list(atom.GetPropNames())[0] != "dummyLabel":
                            bond_to_backbone_idx = atom.GetUnsignedProp("og_idx")
                            # find matching idx from all_dummy_idx
                            result_list_tuples = [
                                idx
                                for idx in all_dummy_idx
                                if bond_to_backbone_idx in idx
                            ]
                            result = [
                                item for tuple in result_list_tuples for item in tuple
                            ]
                            result.remove(bond_to_backbone_idx)

                            bond_to_func_idx = result[0]
                            # NOTE: assumes that edge groups do not have cyclic functional groups attached
                            for f_mol in func_grp:
                                for atom in f_mol.GetAtoms():
                                    if len(list(atom.GetPropNames())) != 0:
                                        if list(atom.GetPropNames())[0] != "dummyLabel":
                                            if (
                                                atom.GetUnsignedProp("og_idx")
                                                == bond_to_func_idx
                                            ):
                                                # join backbone w/ functional grp
                                                if data_aug_bool:
                                                    # remove tuple pairing for augmenting data
                                                    all_augment_idx.remove(
                                                        result_list_tuples[0]
                                                    )
                                                combine_idx = []
                                                combined_mol = rdmolops.CombineMols(
                                                    mol, f_mol
                                                )
                                                for c_atom in combined_mol.GetAtoms():
                                                    if (
                                                        len(list(c_atom.GetPropNames()))
                                                        != 0
                                                    ):
                                                        if (
                                                            list(c_atom.GetPropNames())[
                                                                0
                                                            ]
                                                            != "dummyLabel"
                                                        ):
                                                            if c_atom.GetUnsignedProp(
                                                                "og_idx"
                                                            ) in [
                                                                bond_to_backbone_idx,
                                                                bond_to_func_idx,
                                                            ]:
                                                                combine_idx.append(
                                                                    c_atom.GetIdx()
                                                                )
                                                combined_edmol = Chem.EditableMol(
                                                    combined_mol
                                                )
                                                combined_edmol.AddBond(
                                                    combine_idx[0],
                                                    combine_idx[1],
                                                    Chem.BondType.SINGLE,
                                                )
                                                mol = combined_edmol.GetMol()
                                                # remove dummy atoms involved in AddBond
                                                dummy_remove_idx = []
                                                # find bond atoms with combine_idx
                                                atom0 = mol.GetAtomWithIdx(
                                                    combine_idx[0]
                                                )
                                                atom1 = mol.GetAtomWithIdx(
                                                    combine_idx[1]
                                                )
                                                # find neighbour dummy atoms
                                                for neighbor in atom0.GetNeighbors():
                                                    if neighbor.GetAtomicNum() == 0:
                                                        dummy_remove_idx.append(
                                                            neighbor.GetIdx()
                                                        )
                                                for neighbor in atom1.GetNeighbors():
                                                    if neighbor.GetAtomicNum() == 0:
                                                        dummy_remove_idx.append(
                                                            neighbor.GetIdx()
                                                        )
                                                final_edmol = Chem.EditableMol(mol)
                                                final_edmol.BeginBatchEdit()
                                                for dummy_idx in dummy_remove_idx:
                                                    final_edmol.RemoveAtom(dummy_idx)
                                                final_edmol.CommitBatchEdit()
                                                mol = final_edmol.GetMol()
                                                func_bool = True

                # clean dummy atoms
                if dummy == True:
                    clean_dummy_idx = []
                    clean_edmol = Chem.EditableMol(mol)
                    for atom in mol.GetAtoms():
                        if atom.GetAtomicNum() == 0:
                            clean_dummy_idx.append(atom.GetIdx())
                    clean_edmol.BeginBatchEdit()
                    for idx in clean_dummy_idx:
                        clean_edmol.RemoveAtom(idx)
                    clean_edmol.CommitBatchEdit()
                    mol = clean_edmol.GetMol()

                # augment data before sorting (training data doesn't require sorting)
                if data_aug_bool:
                    frag_augment.append(mol)

                # add fragments to list
                if func_bool == False:
                    edge_bool = False
                    for m_mol in methyl_grp:
                        for atom in m_mol.GetAtoms():
                            if len(list(atom.GetPropNames())) != 0:
                                if atom.GetUnsignedProp("og_idx") == bond_to_func_idx:
                                    # categorize as edge group
                                    edge_fragments.append(Chem.MolToSmiles(mol))
                                    edge_bool = True
                    if edge_bool == False:
                        backbone_fragments.append(Chem.MolToSmiles(mol))
                else:
                    # save combined mol
                    core_fragments.append(Chem.MolToSmiles(mol))

            # NOTE: do I want to add _PAD to each category of fragments for feature importance?
            # add fragments to dataframe
            frag_list = copy.copy(backbone_fragments)
            frag_list.extend(core_fragments)
            frag_list.extend(edge_fragments)

            self.data.at[index, "Donor_Fragments"] = frag_list
            self.data.at[index, "Donor_Backbone_Fragments"] = backbone_fragments
            self.data.at[index, "Donor_Core_Fragments"] = core_fragments
            self.data.at[index, "Donor_Edge_Fragments"] = edge_fragments
            # augment fragments by changing starting fragment
            if data_aug_bool:
                self.data.at[
                    index, "Donor_SMILES_augmented"
                ] = self.augment_donor_polymer(frag_augment, all_augment_idx)

            # # check if donor is already in donor frag data
            # if donor_label_data[index] not in unique_donor_len:
            #     donor_len_df.at[index, "Donor_Label"] = donor_label_data[index]
            #     donor_len_df.at[index, "Donor_Num_Frags"] = len(frag_list)
            #     unique_donor_len.append(donor_label_data[index])
            #     # visualize fragments
            #     if len(frag_list) != 6:
            #         print(len(frag_list), donor_label_data[index])
            #         Draw.ShowMol(frag_mol, size=(500, 500), kekulize=False)
            #         break

            # for frag in frag_list: # check if valid SMILES
            #     print(Chem.MolFromSmiles(frag))

            # add label that belongs to fragment
            # add unique frag to donor_dict
            label = donor_label_data[index]
            for frag in frag_list:
                if frag not in donor_dict:
                    donor_dict[frag] = [1, label]
                elif frag in donor_dict:
                    donor_dict[frag][0] += 1
                    # if label not in donor_dict[frag]: # find all donors with this fragment
                    #     donor_dict[frag].append(label)

            # Streamline outputs --> draw input mol and output fragments
            # done in Juypter Notebook?
            # mols = [donor_mol]
            # legend = ["donor"]
            # for frag in backbone_fragments:
            #     mols.append(Chem.MolFromSmiles(frag))
            #     legend.append("backbone")
            # for frag in core_fragments:
            #     mols.append(Chem.MolFromSmiles(frag))
            #     legend.append("core")
            # for frag in edge_fragments:
            #     mols.append(Chem.MolFromSmiles(frag))
            #     legend.append("edge")
            # img = Draw.MolsToGridImage(mols, legends=legend, returnPNG=True)
            # png = img.data
            # with open(IMAGE_PATH + "output.png", "wb+") as outf:
            #     outf.write(png)
            # # ask for input
            # output_result = input("Do fragments match?:")
            # if output_result == "0":  # bad
            #     bad_label.append(label)
            # rally all "bad" outputs

            index += 1

        # print("IMPROPER FRAGMENT LABELS: ", bad_label)

        self.data.to_csv(frag_data, index=False)

        # save distribution of number of fragments
        donor_len_df.to_csv(num_of_frag_data, index=False)

        # return ranked donors (by highest-lowest occurrence)
        donor_tuples = sorted(donor_dict.items(), key=lambda x: x[1], reverse=True)
        donor_dict_ranked = {}
        for tuple in donor_tuples:
            donor_dict_ranked[tuple[0]] = tuple[1]

        return donor_dict_ranked

    def acceptor_frag(self, frag_data, num_of_frag_data):
        """
        Fragmenting molecules based on the acceptor molecules.

        Args:
            frag_data: path to final .csv file for fragmented data
            num_of_frag_data: distribution of fragments per molecule
        
        Returns:
            frag_master_opv_ml.csv: New column with list of fragments, sorted into categories.
            Dictionary containing all unique fragments with number of occurrences in the dataset.
        """
        acceptor_dict = {}  # tracks unique acceptor fragments

        acceptor_smiles_data = self.data["Acceptor_SMILES"]
        acceptor_label_data = self.data["Acceptor"]

        self.data.insert(17, "Acceptor_Fragments", " ")
        self.data.insert(18, "Acceptor_Spacer_Fragments", " ")
        self.data.insert(19, "Acceptor_Core_Fragments", " ")
        self.data.insert(20, "Acceptor_Core_Functional_Fragments", " ")
        self.data.insert(21, "Acceptor_Terminal_Fragments", " ")

        # STATS: documenting distribution of number of lengths
        acc_len_df = pd.read_csv(num_of_frag_data)
        unique_acc_len = (
            []
        )  # keeping track of unique acceptors for length of frag distribution

        acc_len_df["Acceptor_Label"] = " "
        acc_len_df["Acceptor_Num_Frags"] = 0

        # iterate through all donor SMILES
        index = 0

        # bug_tester = []
        # swap between acceptor_smiles_data and acceptor_smiles_frag_data
        for acceptor in acceptor_smiles_data:
            acceptor_mol = Chem.MolFromSmiles(acceptor)
            # add indices to molecule
            # for atom in acceptor_mol.GetAtoms():
            #     atom.SetAtomMapNum(atom.GetIdx())
            # fragment molecule by bond indices
            # NOTE: needs modification for acceptor molecules
            # fragment conditions:
            # atom degree = 3 (bond-type independent)
            # bond type = SINGLE
            bond_idx_list = []
            all_dummy_idx = []
            for bond in acceptor_mol.GetBonds():
                if (
                    str(bond.GetBondType()) == "SINGLE" and bond.IsInRing() == False
                ):  # single bonds between groups
                    idx0 = bond.GetBeginAtomIdx()
                    idx1 = bond.GetEndAtomIdx()
                    atom0 = acceptor_mol.GetAtomWithIdx(idx0)
                    atom1 = acceptor_mol.GetAtomWithIdx(idx1)
                    if (
                        atom0.GetAtomicNum() == 6
                        and atom1.GetAtomicNum() == 6
                        and (
                            (atom0.GetDegree() == 2 and atom1.GetDegree() == 3)
                            or (atom0.GetDegree() == 3 and atom1.GetDegree() == 2)
                            or (atom0.GetDegree() == 3 and atom1.GetDegree() == 3)
                        )
                    ):
                        # check if neighboring bond for 2 degree atom is double bonded
                        if atom0.GetDegree() == 2:
                            bonds = atom0.GetBonds()
                            for b in bonds:
                                if str(b.GetBondType()) == "DOUBLE":
                                    bond_idx = bond.GetIdx()
                                    bond_idx_list.append(bond_idx)
                                    all_dummy_idx.append(
                                        (idx0, idx1)
                                    )  # atom index of broken bonds
                                    # set original index property for atoms from broken bonds
                                    # if atom has > 1 broken bonds, we have to set multiple unsigned prop
                                    # NOTE: cannot use list
                                    if len(list(atom0.GetPropNames())) != 0:
                                        atom0.SetUnsignedProp("og_idx1", idx0)
                                    else:
                                        atom0.SetUnsignedProp("og_idx", idx0)
                                    if len(list(atom1.GetPropNames())) != 0:
                                        atom1.SetUnsignedProp("og_idx1", idx1)
                                    else:
                                        atom1.SetUnsignedProp("og_idx", idx1)
                        elif atom1.GetDegree() == 2:
                            bonds = atom1.GetBonds()
                            for b in bonds:
                                if str(b.GetBondType()) == "DOUBLE":
                                    bond_idx = bond.GetIdx()
                                    bond_idx_list.append(bond_idx)
                                    all_dummy_idx.append(
                                        (idx0, idx1)
                                    )  # atom index of broken bonds
                                    # set original index property for atoms from broken bonds
                                    if len(list(atom0.GetPropNames())) != 0:
                                        atom0.SetUnsignedProp("og_idx1", idx0)
                                    else:
                                        atom0.SetUnsignedProp("og_idx", idx0)
                                    if len(list(atom1.GetPropNames())) != 0:
                                        atom1.SetUnsignedProp("og_idx1", idx1)
                                    else:
                                        atom1.SetUnsignedProp("og_idx", idx1)
                        else:
                            bond_idx = bond.GetIdx()
                            bond_idx_list.append(bond_idx)
                            all_dummy_idx.append(
                                (idx0, idx1)
                            )  # atom index of broken bonds
                            # set original index property for atoms from broken bonds
                            if len(list(atom0.GetPropNames())) != 0:
                                atom0.SetUnsignedProp("og_idx1", idx0)
                            else:
                                atom0.SetUnsignedProp("og_idx", idx0)
                            if len(list(atom1.GetPropNames())) != 0:
                                atom1.SetUnsignedProp("og_idx1", idx1)
                            else:
                                atom1.SetUnsignedProp("og_idx", idx1)
                elif (
                    str(bond.GetBondType()) == "SINGLE" and bond.IsInRing() == True
                ):  # single bonds in cyclic rings
                    idx0 = bond.GetBeginAtomIdx()
                    idx1 = bond.GetEndAtomIdx()
                    atom0 = acceptor_mol.GetAtomWithIdx(idx0)
                    atom1 = acceptor_mol.GetAtomWithIdx(idx1)
                    # print(atom0.GetIsAromatic(), atom1.GetIsAromatic())
                    degree_list = [3, 4]
                    if (
                        atom0.GetDegree() in degree_list
                        and atom1.GetDegree() in degree_list
                    ):
                        # check if neighboring bonds have double bonds
                        bonds0 = atom0.GetBonds()
                        bonds1 = atom1.GetBonds()
                        bond_type_list = []
                        # check neighboring bonds
                        for b0 in bonds0:
                            bond_type_list.append(str(b0.GetBondType()))
                        for b1 in bonds1:
                            bond_type_list.append(str(b1.GetBondType()))
                        # check neighboring atoms and their bonds
                        for atom in atom0.GetNeighbors():
                            for b_n in atom.GetBonds():
                                bond_type_list.append(str(b_n.GetBondType()))
                        for atom in atom1.GetNeighbors():
                            for b_n in atom.GetBonds():
                                bond_type_list.append(str(b_n.GetBondType()))
                        if "DOUBLE" not in bond_type_list:
                            bond_idx = bond.GetIdx()
                            bond_idx_list.append(bond_idx)
                            all_dummy_idx.append(
                                (idx0, idx1)
                            )  # atom index of broken bonds
                            # set original index property for atoms from broken bonds
                            if len(list(atom0.GetPropNames())) != 0:
                                atom0.SetUnsignedProp("og_idx1", idx0)
                            else:
                                atom0.SetUnsignedProp("og_idx", idx0)
                            if len(list(atom1.GetPropNames())) != 0:
                                atom1.SetUnsignedProp("og_idx1", idx1)
                            else:
                                atom1.SetUnsignedProp("og_idx", idx1)

            frag_mol = rdmolops.FragmentOnBonds(
                acceptor_mol, bond_idx_list, addDummies=True
            )
            fragments = []
            # frag_smiles = Chem.MolToSmiles(frag_mol)
            # frag_list = frag_smiles.split(".")

            terminal_grp = []
            spacer_grp = []
            core_grp = []
            core_func_grp = []
            list_frag_mols = list(rdmolops.GetMolFrags(frag_mol, asMols=True))

            # sort fragments into corresponding groups
            for mol in list_frag_mols:
                n_dummy = 0
                core_func_bool = False
                for atom in mol.GetAtoms():
                    if len(list(atom.GetPropNames())) != 0:
                        if list(atom.GetPropNames())[0] != "dummyLabel":
                            if len(list(atom.GetPropNames())) == 2:
                                n_dummy += 2
                                core_func_bool = True
                            else:
                                n_dummy += 1
                if n_dummy == 1:
                    # remove all dummy atoms
                    rw_mol = Chem.RWMol(mol)
                    rw_mol.BeginBatchEdit()
                    for rw_atom_dummy in rw_mol.GetAtoms():
                        if rw_atom_dummy.GetAtomicNum() == 0:
                            rw_mol.RemoveAtom(rw_atom_dummy.GetIdx())
                    rw_mol.CommitBatchEdit()
                    mol = rw_mol.GetMol()
                    terminal_grp.append(Chem.MolToSmiles(mol))
                    # terminal_grp.append(mol)
                elif n_dummy == 2 and core_func_bool:
                    # remove all dummy atoms
                    rw_mol = Chem.RWMol(mol)
                    rw_mol.BeginBatchEdit()
                    for rw_atom_dummy in rw_mol.GetAtoms():
                        if rw_atom_dummy.GetAtomicNum() == 0:
                            rw_mol.RemoveAtom(rw_atom_dummy.GetIdx())
                    rw_mol.CommitBatchEdit()
                    mol = rw_mol.GetMol()
                    core_func_grp.append(Chem.MolToSmiles(mol))
                    # core_func_grp.append(mol)
                elif n_dummy == 2 and not core_func_bool:
                    # remove all dummy atoms
                    rw_mol = Chem.RWMol(mol)
                    rw_mol.BeginBatchEdit()
                    for rw_atom_dummy in rw_mol.GetAtoms():
                        if rw_atom_dummy.GetAtomicNum() == 0:
                            rw_mol.RemoveAtom(rw_atom_dummy.GetIdx())
                    rw_mol.CommitBatchEdit()
                    mol = rw_mol.GetMol()
                    spacer_grp.append(Chem.MolToSmiles(mol))
                    # spacer_grp.append(mol)
                elif n_dummy == 3:
                    # remove all dummy atoms
                    rw_mol = Chem.RWMol(mol)
                    rw_mol.BeginBatchEdit()
                    for rw_atom_dummy in rw_mol.GetAtoms():
                        if rw_atom_dummy.GetAtomicNum() == 0:
                            rw_mol.RemoveAtom(rw_atom_dummy.GetIdx())
                    rw_mol.CommitBatchEdit()
                    mol = rw_mol.GetMol()
                    spacer_grp.append(Chem.MolToSmiles(mol))
                    # spacer_grp.append(mol)
                elif n_dummy == 4:
                    # remove all dummy atoms
                    rw_mol = Chem.RWMol(mol)
                    rw_mol.BeginBatchEdit()
                    for rw_atom_dummy in rw_mol.GetAtoms():
                        if rw_atom_dummy.GetAtomicNum() == 0:
                            rw_mol.RemoveAtom(rw_atom_dummy.GetIdx())
                    rw_mol.CommitBatchEdit()
                    mol = rw_mol.GetMol()
                    core_grp.append(Chem.MolToSmiles(mol))
                    # core_grp.append(mol)

                fragments.append(Chem.MolToSmiles(mol))
            # add label to acceptor_dict
            # add unique frag to acceptor_dict
            frag_list = copy.copy(spacer_grp)
            frag_list.extend(core_grp)
            frag_list.extend(core_func_grp)
            frag_list.extend(terminal_grp)
            label = acceptor_label_data[index]
            for frag in frag_list:
                if frag not in acceptor_dict:
                    acceptor_dict[frag] = [1, label]
                elif frag in acceptor_dict:
                    acceptor_dict[frag][0] += 1

            if acceptor_label_data[index] not in unique_acc_len:
                acc_len_df.at[index, "Acceptor_Label"] = acceptor_label_data[index]
                acc_len_df.at[index, "Acceptor_Num_Frags"] = len(frag_list)
                unique_acc_len.append(acceptor_label_data[index])
                # visualize fragments
                # if len(frag_list) != 7:
                # print(len(frag_list), acceptor_label_data[index])
                # Draw.ShowMol(frag_mol, size=(500, 500), kekulize=False)

            # add fragments to dataframe
            self.data.at[index, "Acceptor_Fragments"] = frag_list
            self.data.at[index, "Acceptor_Spacer_Fragments"] = spacer_grp
            self.data.at[index, "Acceptor_Core_Fragments"] = core_grp
            self.data.at[index, "Acceptor_Core_Functional_Fragments"] = core_func_grp
            self.data.at[index, "Acceptor_Terminal_Fragments"] = terminal_grp
            # if index == 144:  # need to fix this issue where the side group is broken
            #     Draw.ShowMol(frag_mol, size=(500, 500), kekulize=False)
            # Draw.MolToFile(
            #     frag_mol, size=(800, 800), filename=IMAGE_PATH + "frag_acceptor_mol.png"
            # )
            # Draw.ShowMol(frag_mol, size=(500, 500), kekulize=False)
            index += 1

        # print(acceptor_dict)

        self.data.to_csv(frag_data, index=False)

        # save distribution of number of fragments
        acc_len_df.to_csv(num_of_frag_data, index=False)

        # return ranked acceptors (by highest-lowest occurrence)
        acceptor_tuples = sorted(
            acceptor_dict.items(), key=lambda x: x[1], reverse=True
        )
        acceptor_dict_ranked = {}
        for tuple in acceptor_tuples:
            acceptor_dict_ranked[tuple[0]] = tuple[1]

        return acceptor_dict_ranked

    def tokenize_frag(self, donor_dict, acceptor_dict, frag_data, train_frag_data):
        """
        Tokenizes the fragments in the dataset

        Args:
            donor_dict: dictionary of unique fragments in donor molecules
            acceptor_dict: dictionary of unique fragments in acceptor molecules
            frag_data: path to final .csv file for fragmented data
            train_frag_data: path to .csv file for training
        
        Returns:
            train_frag_master.csv: New column with list of fragments, sorted into categories, and (optionally) augmented.
            Dictionary containing all unique fragments with number of occurrences in the dataset.
        """
        unique_token_dict = (
            {}
        )  # contains all unique fragments in donor and acceptor molecules
        unique_token_dict["_PAD"] = 0
        unique_token_dict["."] = 1
        id = len(unique_token_dict)
        for frag in donor_dict:
            unique_token_dict[frag] = id
            id += 1

        for frag in acceptor_dict:
            if frag in list(unique_token_dict.keys()):
                continue
            else:
                unique_token_dict[frag] = id
                id += 1
        print("VOCAB LENGTH: ", len(unique_token_dict.keys()))
        frag_df = pd.read_csv(frag_data)

        # create new dataframe for training only
        column_names = [
            "Donor",
            "Donor_SMILES",
            "Donor_SELFIES",
            "Acceptor",
            "Acceptor_SMILES",
            "Acceptor_SELFIES",
            "DA_pair_fragments",
            "AD_pair_fragments",
            "DA_pair_tokenized",
            "AD_pair_tokenized",
            "DA_pair_tokenized_aug",
            "AD_pair_tokenized_aug",
            "PCE(%)",
        ]
        train_frag_df = pd.DataFrame(columns=column_names)
        train_frag_df["Donor"] = frag_df["Donor"]
        train_frag_df["Donor_SMILES"] = frag_df["Donor_SMILES"]
        train_frag_df["Donor_SELFIES"] = frag_df["Donor_SELFIES"]
        train_frag_df["Acceptor"] = frag_df["Acceptor"]
        train_frag_df["Acceptor_SMILES"] = frag_df["Acceptor_SMILES"]
        train_frag_df["Acceptor_SELFIES"] = frag_df["Acceptor_SELFIES"]
        train_frag_df["PCE(%)"] = frag_df["PCE(%)"]

        # combine DA pair fragments
        donor_frags = frag_df["Donor_Fragments"]
        acceptor_frags = frag_df["Acceptor_Fragments"]

        # augmented Donor fragments
        donor_frags_aug = frag_df["Donor_SMILES_augmented"]

        # find max sequence length for donor and acceptor
        index = 0
        donor_max_seq = 0
        acceptor_max_seq = 0
        max_seq = 0
        while index < len(frag_df):
            donor_len = len(ast.literal_eval(donor_frags[index]))
            acceptor_len = len(ast.literal_eval(acceptor_frags[index]))
            if donor_len > donor_max_seq:
                donor_max_seq = donor_len
            if acceptor_len > acceptor_max_seq:
                acceptor_max_seq = acceptor_len
            if donor_len + acceptor_len + 1 > max_seq:
                max_seq = donor_len + acceptor_len + 1
            index += 1

        print(max_seq)

        # iterate through dataframe
        index = 0
        while index < len(frag_df):
            # add (pre)padding to fragment lists
            # adds padding to donor and padding to acceptor so individually they are same length
            # lengths are also the same (the "." is always in same place)
            # donor_frag_list = ast.literal_eval(donor_frags[index])
            # if len(donor_frag_list) < donor_max_seq:
            #     for i in range(donor_max_seq - len(donor_frag_list)):
            #         donor_frag_list.insert(0, "_PAD")

            # # create copy of donor_frag_list after padding for augmentation
            # donor_frag_list_copy = copy.copy(donor_frag_list)

            # acceptor_frag_list = ast.literal_eval(acceptor_frags[index])
            # if len(acceptor_frag_list) < acceptor_max_seq:
            #     for i in range(acceptor_max_seq - len(acceptor_frag_list)):
            #         acceptor_frag_list.insert(0, "_PAD")

            # adds padding only to the beginning of the entire D.A pair
            donor_frag_list = ast.literal_eval(donor_frags[index])
            acceptor_frag_list = ast.literal_eval(acceptor_frags[index])
            donor_frag_list_copy = copy.copy(donor_frag_list)

            # add "." as separator between donor and acceptor
            donor_frag_list.append(".")
            # concatenate donor_frag and acceptor_frag lists
            donor_frag_list.extend(acceptor_frag_list)
            if len(donor_frag_list) < max_seq:
                for i in range(max_seq - len(donor_frag_list)):
                    donor_frag_list.insert(0, "_PAD")
            train_frag_df.at[index, "DA_pair_fragments"] = donor_frag_list

            # augment data
            acceptor_frag_list.append(".")
            acceptor_frag_list.extend(donor_frag_list_copy)
            if len(acceptor_frag_list) < max_seq:
                for i in range(max_seq - len(acceptor_frag_list)):
                    acceptor_frag_list.insert(0, "_PAD")
            train_frag_df.at[index, "AD_pair_fragments"] = acceptor_frag_list

            # convert fragments into tokenized numbers
            tokenized_list = []
            for frag in donor_frag_list:
                tokenized_list.append(unique_token_dict[frag])
            train_frag_df.at[index, "DA_pair_tokenized"] = tokenized_list

            # convert augmented fragments into tokenized numbers
            tokenized_list = []
            for frag in acceptor_frag_list:
                tokenized_list.append(unique_token_dict[frag])
            train_frag_df.at[index, "AD_pair_tokenized"] = tokenized_list
            index += 1

        # DONOR AUGMENTED DATA
        # iterate through dataframe
        index = 0
        num_of_aug = 0  # to track the number of augmented fragment data
        while index < len(frag_df):
            da_pair_aug = []
            ad_pair_aug = []
            da_token_aug = []
            ad_token_aug = []
            donor_frags_aug_list = ast.literal_eval(donor_frags_aug[index])
            for donor_aug in donor_frags_aug_list:
                donor_frag_list = donor_aug
                # if len(donor_frag_list) < donor_max_seq:
                #     for i in range(donor_max_seq - len(donor_frag_list)):
                #         donor_frag_list.insert(0, "_PAD")

                # # create copy of donor_frag_list after padding for augmentation
                # donor_frag_list_copy = copy.copy(donor_frag_list)

                # acceptor_frag_list = ast.literal_eval(acceptor_frags[index])
                # if len(acceptor_frag_list) < acceptor_max_seq:
                #     for i in range(acceptor_max_seq - len(acceptor_frag_list)):
                #         acceptor_frag_list.insert(0, "_PAD")

                # adds padding only to the beginning of the entire D.A pair
                donor_frag_list_copy = copy.copy(donor_frag_list)
                acceptor_frag_list = ast.literal_eval(acceptor_frags[index])
                # add "." as separator between donor and acceptor
                donor_frag_list.append(".")
                # concatenate donor_frag and acceptor_frag lists
                donor_frag_list.extend(acceptor_frag_list)
                if len(donor_frag_list) < max_seq:
                    for i in range(max_seq - len(donor_frag_list)):
                        donor_frag_list.insert(0, "_PAD")
                da_pair_aug.append(donor_frag_list)

                # augment data
                acceptor_frag_list.append(".")
                acceptor_frag_list.extend(donor_frag_list_copy)
                if len(acceptor_frag_list) < max_seq:
                    for i in range(max_seq - len(acceptor_frag_list)):
                        acceptor_frag_list.insert(0, "_PAD")
                ad_pair_aug.append(acceptor_frag_list)

                # convert fragments into tokenized numbers
                tokenized_list = []
                for frag in donor_frag_list:
                    tokenized_list.append(unique_token_dict[frag])
                da_token_aug.append(tokenized_list)

                # convert augmented fragments into tokenized numbers
                tokenized_list = []
                for frag in acceptor_frag_list:
                    tokenized_list.append(unique_token_dict[frag])
                ad_token_aug.append(tokenized_list)
                num_of_aug += 1
            train_frag_df.at[index, "DA_pair_tokenized_aug"] = da_token_aug
            train_frag_df.at[index, "AD_pair_tokenized_aug"] = ad_token_aug
            index += 1
        train_frag_df.to_csv(train_frag_data)

        # NOTE: STATS FOR AUGMENTATION
        # NO AUGMENTATION = 443
        # DA+AD = 886
        # D1A+AD1, D2A+AD2... = num_of_aug * 2 = 3154

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
            molsPerRow=16,
            maxMols=200,
            subImgSize=(300, 300),
            legends=frag_legends,
        )
        display(img)

    def plot_num_frag(self, num_of_frag_data):
        """
        Function that plots the histogram/distribution of number of fragments 

        Args:
            num_of_frag_data: number of fragments per molecule
        
        Returns:
            plot of distribution of number of fragments per molecule
        """
        n_frag_df = pd.read_csv(num_of_frag_data)
        print(n_frag_df.head())
        ax = n_frag_df.plot.hist(alpha=0.5)
        plt.show()

    def frag_substruct_filter(self, substruct_smi, frag_dict):
        """
        Function that visualizes all the fragments with a defined substructure

        Args:
            substruct_smi: SMILES of the substruct you want to filter
            frag_dict: dictionary of unique fragments from donor and acceptor molecules
        
        Returns:
            filtered_dict: dictionary of fragments filtered by the presence of the input substructure
        """
        substruct_mol = Chem.MolFromSmiles(substruct_smi)
        filtered_dict = {}
        for frag_key in frag_dict.keys():
            frag_mol_key = Chem.MolFromSmiles(frag_key)
            if frag_mol_key is not None:
                if frag_mol_key.HasSubstructMatch(substruct_mol):
                    filtered_dict[frag_key] = frag_dict[frag_key]

        return filtered_dict


frag = Fragger(MASTER_DATA)
d_dict = frag.donor_frag(FRAG_MASTER_DATA, NUM_FRAG_DATA, True)
a_dict = frag.acceptor_frag(FRAG_MASTER_DATA, NUM_FRAG_DATA)
frag.tokenize_frag(d_dict, a_dict, FRAG_MASTER_DATA, TRAIN_MASTER_DATA)

# FILTER for SUBSTRUCTURE in FRAGMENTS
# substruct_indane = "C1CC2=CC=CC=C2C1"
# substruct_benzodithiophene = "c3cc2cc1sccc1cc2s3"
# a_substruct_dict = a_frag.frag_substruct_filter(substruct_indane, a_dict)
# d_substruct_dict = frag.frag_substruct_filter(substruct_benzodithiophene, d_dict)

# VISUALIZATION
# Wu et. al paper: donor_frags = 86, acceptor_frags = 111
# My fragmentation: donor_frags = 128, acceptor_frags = 127 or 129 (w/o atomdegree0,1 = 2,2)
# My fragmentation (w/o R grp): donor_frags = 73, acceptor_frags = 95
# frag.frag_visualization(a_dict)
# frag.frag_visualization(d_dict)

# plot num of fragment distribution
# frag.plot_num_frag(NUM_FRAG_DATA)

