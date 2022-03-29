from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolDrawing, DrawingOptions
import pkg_resources
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, rdmolops
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import display
from collections import deque
import numpy as np
import copy
import time

IMAGE_PATH = pkg_resources.resource_filename("opv_ml", "data/postprocess/test/")


def remove_methyl(smi):
    """
        Function that checks the number of methyl groups and removes them on donor molecules.
        Only the starting and ending methyl group are deleted.
        """
    donor_mol = Chem.MolFromSmiles(smi)
    n_methyl = 0
    donor_edmol = Chem.EditableMol(donor_mol)
    remove_idx = []
    for atom in donor_edmol.GetMol().GetAtoms():
        if atom.GetDegree() == 1 and atom.GetAtomicNum() == 6:
            for neighbour in atom.GetNeighbors():
                if neighbour.GetIsAromatic():
                    n_methyl += 1
                    atom_idx = atom.GetIdx()
                    remove_idx.append(atom_idx)

    # performs action all at once so index doesn't change
    donor_edmol.BeginBatchEdit()
    for idx in remove_idx:
        donor_edmol.RemoveAtom(idx)
    donor_edmol.CommitBatchEdit()

    # Draw.ShowMol(donor_mol, size=(600, 600))
    # Draw.ShowMol(donor_edmol.GetMol(), size=(600, 600))
    donor_smi = Chem.MolToSmiles(donor_edmol.GetMol())
    return donor_edmol.GetMol()


def attach_all_func_to_core(mol, backbone_grp, all_dummy_idx, old_bond_idx):
    """
    Function that checks whether all functional groups are attached to core.
    If the incoming mol has 1 dummy atom, then there is more functional group attachment to be done.
    Test Case
    -----------
    smi = "Cc1ccc(c2c3nn(SCC(CCCCCC)CCCC)nc3c(c3ccc(c4sc5c(c6ccc(c7ccc(OCC(CCCCCC)CCCCCCCC)cc7)cc6)c6cc(C)sc6c(c6ccc(c7ccc(OCC(CCCCCC)CCCCCCCC)cc7)cc6)c5c4)s3)c(F)c2F)s1"
    """
    # print(Chem.MolToSmiles(mol))
    # check number of dummy atoms
    num_dummy = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            num_dummy += 1

    # get bond index of the one dummy atom
    all_results = []
    if num_dummy == 1:
        for atom in mol.GetAtoms():
            if len(list(atom.GetPropNames())) != 0:
                bond_og_idx = atom.GetUnsignedProp("og_idx")
                # find matching idx from all_dummy_idx
                result_list_tuples = [
                    idx for idx in all_dummy_idx if bond_og_idx in idx
                ]
                all_results.extend(result_list_tuples)

        # remove all old bonds from all available bond indexes
        # keeps only new bonds
        all_results = [
            idx_tuple
            for idx_tuple in all_results
            if sorted(idx_tuple) != sorted(old_bond_idx)
        ]

        # converts list of tuples into list
        all_results = [item for t in all_results for item in t]

        for backbone_mol in backbone_grp:
            for b_atom in backbone_mol.GetAtoms():
                if len(list(b_atom.GetPropNames())) != 0:
                    atom_og_idx = b_atom.GetUnsignedProp("og_idx")
                    if atom_og_idx in all_results:

                        combined_mol = rdmolops.CombineMols(mol, backbone_mol)
                        combine_idx = []
                        for c_atom in combined_mol.GetAtoms():
                            if len(list(c_atom.GetPropNames())) != 0:
                                if (
                                    c_atom.GetUnsignedProp("og_idx") == all_results[0]
                                    or c_atom.GetUnsignedProp("og_idx")
                                    == all_results[1]
                                ):
                                    combine_idx.append(c_atom.GetIdx())
                        combined_edmol = Chem.EditableMol(combined_mol)
                        combined_edmol.AddBond(
                            combine_idx[0], combine_idx[1], Chem.BondType.SINGLE,
                        )
                        mol = combined_edmol.GetMol()
                        # remove dummy atoms involved in AddBond
                        dummy_remove_idx = []
                        # find bond atoms with combine_idx
                        atom0 = mol.GetAtomWithIdx(combine_idx[0])
                        atom1 = mol.GetAtomWithIdx(combine_idx[1])
                        # find neighbour dummy atoms
                        for neighbor in atom0.GetNeighbors():
                            if neighbor.GetAtomicNum() == 0:
                                dummy_remove_idx.append(neighbor.GetIdx())
                        for neighbor in atom1.GetNeighbors():
                            if neighbor.GetAtomicNum() == 0:
                                dummy_remove_idx.append(neighbor.GetIdx())
                        final_edmol = Chem.EditableMol(mol)
                        final_edmol.BeginBatchEdit()
                        for dummy_idx in dummy_remove_idx:
                            final_edmol.RemoveAtom(dummy_idx)
                        final_edmol.CommitBatchEdit()
                        mol = final_edmol.GetMol()
    return mol


def augment_donor_polymer(donor_frag_list, bond_idx_list):
    """
    Function that returns list of data augmented donor fragments in SMILES

    Input
    ---------
    donor_frag_list: list of unordered donor fragments created from donor molecule (1-4 fragments)

    Return
    ---------
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
                result_list_tuples = [idx for idx in bond_idx_list if bond_idx in idx]
                result = [item for tuple in result_list_tuples for item in tuple]
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
                print(bond_idx, other_bond_idx_list)
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


# smi1 = "c1ccccc1"
# smi2 = "c1ccccc1"
# smi3 = "c2ccc1ccccc1c2"
# mol1 = Chem.MolFromSmiles(smi1)
# mol2 = Chem.MolFromSmiles(smi2)
# mol3 = Chem.rdmolops.CombineMols(mol1, mol2)
# edmol = Chem.EditableMol(mol3)
# edmol.RemoveBond(7, 8)
# edmol.AddBond(4, 8, order=Chem.rdchem.BondType.AROMATIC)
# edmol.AddBond(5, 7, order=Chem.rdchem.BondType.AROMATIC)
# back = edmol.GetMol()


# mol4 = Chem.MolFromSmiles(smi3)
# print(Chem.MolToSmiles(mol3))
# for atom in mol3.GetAtoms():
#     atom.SetAtomMapNum(atom.GetIdx())
# for atom in back.GetAtoms():
#     atom.SetAtomMapNum(atom.GetIdx())
# # Draw.MolToFile(mol3, size=(800, 800), filename=IMAGE_PATH + "combine_frag.png")
# Draw.MolToFile(back, size=(800, 800), filename=IMAGE_PATH + "edited_combine_frag.png")


# NOTE: what props are available
# props = rdkit.RDProps()
# print(props.getPropList())

# smi = "CCCCC(CC)COC(=O)c1c(-c2cccs2)sc2c(C(=O)OCC(CC)CCCC)c(-c3ccc(-c4cc5c(-c6ccc(CC(CC)CCCC)s6)c6sccc6c(-c6ccc(CC(CC)CCCC)s6)c5s4)s3)sc12"
# smi = "CCCCC(CC)COC(=O)c1c(-c2ccc(C)s2)sc2c(C(=O)OCC(CC)CCCC)c(-c3ccc(-c4cc5c(-c6ccc(CC(CC)CCCC)s6)c6sc(C)cc6c(-c6ccc(CC(CC)CCCC)s6)c5s4)s3)sc12"
# smi = "Cc1ccc(c2c3nn(CC(CCCCCC)CCCCCCCC)nc3c(c3ccc(c4sc5c(c6ccc(c7ccc(SCC(CCCCCC)CCCCCCCC)cc7)cc6)c6cc(C)sc6c(c6ccc(c7ccc(SCC(CCCCCC)CCCCCCCC)cc7)cc6)c5c4)s3)c(F)c2F)s1"
# smi = "*c1ccc(-c2c3cc(-c4ccc(-c5sc(-c6ccc(C)s6)c6c5C(=O)c5c(*)sc(*)c5C6=O)s4)sc3c(-c3ccc(*)s3)c3cc(C)sc23)s1"
smi = "Cc1ccc(c2c3nn(SCC(CCCCCC)CCCC)nc3c(c3ccc(c4sc5c(c6ccc(c7ccc(OCC(CCCCCC)CCCCCCCC)cc7)cc6)c6cc(C)sc6c(c6ccc(c7ccc(OCC(CCCCCC)CCCCCCCC)cc7)cc6)c5c4)s3)c(F)c2F)s1"
donor_mol = Chem.MolFromSmiles(smi)

data_aug_bool = True

# for atom in donor_mol.GetAtoms():
#     atom.SetAtomMapNum(atom.GetIdx())

Draw.MolToFile(
    donor_mol, size=(800, 800), filename=IMAGE_PATH + "test_donor_mol.png",
)
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

frag_mol = rdmolops.FragmentOnBonds(donor_mol, bond_idx_list, addDummies=True)

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

# print("BACK: ", backbone_grp)
# print("FUNC: ", func_grp)
# print("METHYL: ", methyl_grp)

backbone_fragments = []
core_fragments = []
edge_fragments = []

frag_augment = []
all_augment_idx = copy.copy(all_dummy_idx)

backbone_grp_copy = copy.copy(backbone_grp)

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
                    idx for idx in all_dummy_idx if bond_to_backbone_idx in idx
                ]
                result = [item for tuple in result_list_tuples for item in tuple]
                result.remove(bond_to_backbone_idx)
                bond_to_func_idx = result[0]
                # NOTE: assumes that edge groups do not have cyclic functional groups attached
                for f_mol in func_grp:
                    for atom in f_mol.GetAtoms():
                        if len(list(atom.GetPropNames())) != 0:
                            if list(atom.GetPropNames())[0] != "dummyLabel":
                                if atom.GetUnsignedProp("og_idx") == bond_to_func_idx:
                                    # join backbone w/ functional grp
                                    # remove tuple pairing for augmenting data
                                    all_augment_idx.remove(result_list_tuples[0])

                                    # remove backbone grp from backbone_grp
                                    backbone_grp_copy.remove(mol)
                                    combine_idx = []
                                    combined_mol = rdmolops.CombineMols(mol, f_mol)
                                    for c_atom in combined_mol.GetAtoms():
                                        if len(list(c_atom.GetPropNames())) != 0:
                                            if (
                                                list(c_atom.GetPropNames())[0]
                                                != "dummyLabel"
                                            ):
                                                if c_atom.GetUnsignedProp("og_idx") in [
                                                    bond_to_backbone_idx,
                                                    bond_to_func_idx,
                                                ]:
                                                    combine_idx.append(c_atom.GetIdx())
                                    combined_edmol = Chem.EditableMol(combined_mol)
                                    combined_edmol.AddBond(
                                        combine_idx[0],
                                        combine_idx[1],
                                        Chem.BondType.SINGLE,
                                    )
                                    mol = combined_edmol.GetMol()
                                    # remove dummy atoms involved in AddBond
                                    dummy_remove_idx = []
                                    # find bond atoms with combine_idx
                                    atom0 = mol.GetAtomWithIdx(combine_idx[0])
                                    atom1 = mol.GetAtomWithIdx(combine_idx[1])
                                    # find neighbour dummy atoms
                                    for neighbor in atom0.GetNeighbors():
                                        if neighbor.GetAtomicNum() == 0:
                                            dummy_remove_idx.append(neighbor.GetIdx())
                                    for neighbor in atom1.GetNeighbors():
                                        if neighbor.GetAtomicNum() == 0:
                                            dummy_remove_idx.append(neighbor.GetIdx())
                                    final_edmol = Chem.EditableMol(mol)
                                    final_edmol.BeginBatchEdit()
                                    for dummy_idx in dummy_remove_idx:
                                        final_edmol.RemoveAtom(dummy_idx)
                                    final_edmol.CommitBatchEdit()
                                    mol = final_edmol.GetMol()
                                    mol = attach_all_func_to_core(
                                        mol,
                                        backbone_grp_copy,
                                        all_dummy_idx,
                                        (bond_to_backbone_idx, bond_to_func_idx),
                                    )
                                    func_bool = True

    # augment data before sorting (training data doesn't require sorting)
    if data_aug_bool:
        frag_augment.append(mol)

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


for frag in backbone_fragments:
    print(frag)

for frag in core_fragments:
    print(frag)

# print("BACKBONE: ", backbone_fragments)
# print("CORE: ", core_fragments)
# print("EDGE: ", edge_fragments)

# augment fragment
# aug = augment_donor_polymer(frag_augment, all_augment_idx)

# find idx of other dummy bond in all_dummy_idx
# look through backbone_grp for molecule with idx of other dummy bond
# connect the two groups via substructure of functional group? NOTE: RDKIT

# remove dummy atoms involved in AddBond
# dummy_remove_idx = []
# for atom in mol.GetAtoms():
#     if (
#         atom.GetAtomicNum() == 0
#     ):  # remove dummy atom from functional grps
#         dummy_remove_idx.append(atom.GetIdx())

# NOTE: Recap doesn't work because its basically just rules but I have limited control and manipulation

# from rdkit.Chem import Recap

# recap_donor_smi = "CCCCC(CC)COC(=O)c1c(-c2cccs2)sc2c(C(=O)OCC(CC)CCCC)c(-c3ccc(-c4cc5c(-c6ccc(CC(CC)CCCC)s6)c6sccc6c(-c6ccc(CC(CC)CCCC)s6)c5s4)s3)sc12"
# recap_mol = Chem.MolFromSmiles(recap_donor_smi)

# hierarch = Recap.RecapDecompose(recap_mol)
# ks = hierarch.GetLeaves().keys()
# recap_frag_smi = ""
# index = 0
# for smile in ks:
#     recap_frag_smi = recap_frag_smi + smile
#     if index != len(ks) - 1:
#         recap_frag_smi = recap_frag_smi + "."
#     index += 1

# print(recap_frag_smi)
# recap_frag_mol = Chem.MolFromSmiles(recap_frag_smi)

# Draw.MolToFile(
#     recap_frag_mol, size=(800, 800), filename=IMAGE_PATH + "test_mol_frag_working.png"
# )

# NOTE: BRICS doesn't work because its basically just rules but I have limited control and manipulation
# Same problem as Recap

# from rdkit.Chem import BRICS

# brics_donor_smi = "CCCCC(CC)COC(=O)c1c(-c2cccs2)sc2c(C(=O)OCC(CC)CCCC)c(-c3ccc(-c4cc5c(-c6ccc(CC(CC)CCCC)s6)c6sccc6c(-c6ccc(CC(CC)CCCC)s6)c5s4)s3)sc12"
# brics_mol = Chem.MolFromSmiles(brics_donor_smi)

# brics_frag_list = sorted(BRICS.BRICSDecompose(brics_mol))
# brics_frag_smi = ""

# index = 0
# for smile in brics_frag_list:
#     brics_frag_smi = brics_frag_smi + smile
#     if index != len(brics_frag_list) - 1:
#         brics_frag_smi = brics_frag_smi + "."
#     index += 1

# print(brics_frag_smi)

# smi = "CC(CCCCCCCCCCCC)CCCCCCCCCCc1c(c2sc(c3sc(C)cc3)cc2)sc(c2c3nsnc3c(c3cc(CC(CCCCCCCCCCCC)CCCCCCCCCC)c(C)s3)c(F)c2F)c1"
# mol = Chem.MolFromSmiles(smi)
# Draw.ShowMol(mol)
# smi_list = []
# start_time = time.time()
# for i in range(10000):
#     random_smi = Chem.MolToSmiles(mol, doRandom=True)
#     if random_smi not in smi_list:
#         smi_list.append(random_smi)


# # 10000 -> 9996
# # 100000 -> 99568 (78s)
# # 1000000 ->

# print(len(smi_list))
# for i in range(0, 99999, 20000):
#     random_mol = Chem.MolFromSmiles(smi_list[i])
#     Draw.ShowMol(random_mol)
