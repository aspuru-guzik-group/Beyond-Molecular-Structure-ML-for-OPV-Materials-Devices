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
import numpy as np
import copy
import itertools

IMAGE_PATH = pkg_resources.resource_filename("opv_ml", "data/postprocess/test/")

# smi = "CCCCCCc1ccc(C2(c3ccc(CCCCCC)cc3)c3c4sc(/C=C5C(c6ccccc6C/5=C(C#N)\C#N)=O)cc4sc3c3cc4c(c5sc6cc(/C=C7C(c8ccccc8C/7=C(C#N)\C#N)=O)sc6c5C(c5ccc(CCCCCC)cc5)4c4ccc(CCCCCC)cc4)cc32)cc1"

# acceptor_mol = Chem.MolFromSmiles(smi)

# Draw.MolToFile(
#     acceptor_mol, size=(800, 800), filename=IMAGE_PATH + "test_acceptor_mol.png",
# )
# acceptor = "O=C1c2ccc(C)cc2C(/C1=C/c1cc2c(c3c4ccc5c(c6sc(/C=C7/C(c8cc(C)ccc8C7=O)=C(C#N)\C#N)cc6C(c6ccc(CCCCCC)cc6)5c5ccc(CCCCCC)cc5)c4ccc3C2(c2ccc(CCCCCC)cc2)c2ccc(CCCCCC)cc2)s1)=C(C#N)/C#N"
# acceptor = "CCCCCCc1ccc(C2(c3ccc(CCCCCC)cc3)c3c4sc(/C=C5C(c6ccccc6C/5=C(C#N)\C#N)=O)cc4sc3c3cc4c(c5sc6cc(/C=C7C(c8ccccc8C/7=C(C#N)\C#N)=O)sc6c5C(c5ccc(CCCCCC)cc5)4c4ccc(CCCCCC)cc4)cc32)cc1"
# acceptor = "O=C(c1cc(F)c(F)cc1C/2=C(C#N)/C#N)C2=C/c3sc4c(C(c5ccc(CCCCCC)cc5)(c6ccc(CCCCCC)cc6)c7c8c(CCCCCC)cc9c%10sc(c%11sc(/C=C%12C(c%13cc(F)c(F)cc%13C\%12=C(C#N)\C#N)=O)cc%11C%14(c%15ccc(CCCCCC)cc%15)c%16ccc(CCCCCC)cc%16)c%14c%10c(CCCCCC)cc9c8sc74)c3"
# acceptor = "CCCCCCc1ccc(C2(c3ccc(CCCCCC)cc3)c3cc(c4c(OCC(CC)CCCC)cc(/C=C5C(c6cc(Cl)c(Cl)cc6C/5=C(C#N)\C#N)=O)s4)sc3c3cc4c(c5sc(c6c(OCC(CC)CCCC)cc(/C=C7C(c8cc(Cl)c(Cl)cc8C/7=C(C#N)\C#N)=O)s6)cc5C(c5ccc(CCCCCC)cc5)4c4ccc(CCCCCC)cc4)cc32)cc1"
# DTPC-DFIC
# acceptor = "O=C(c%47cc(F)c(F)cc%47C/%48=C(C#N)/C#N)C%48=C/c%49sc(c(cc(c%50c%51%52)n%53CC(CCCCCC)CCCCCCCC)c%52c(C(c%54ccc(CCCCCC)cc%54)%55c%56ccc(CCCCCC)cc%56)ccc%51c%57c%58c%50c%53cc%59c%58c(C(c%60ccc(CCCCCC)cc%60)(c%61ccc(CCCCCC)cc%61)c%62c%59sc(/C=C%63/C(c%64c(C%63=O)cc(F)c(F)c%64)=C(C#N)/C#N)c%62)cc%57)c%55c%49"
# acceptor = "CC(CCCCCC)CCCCCCCCC1(CC(CCCCCC)CCCCCCCC)c2cc3c(Cc4cc(/C=C5C(c6c(C/5=C(C#N)\C#N)ccc(F)c6)=O)sc43)cc2C2C(CC(CCCCCC)CCCCCCCC)(CC(CCCCCC)CCCCCCCC)c3cc4c(Cc5cc(/C=C6C(c7c(C/6=C(C#N)\C#N)ccc(F)c7)=O)sc54)cc3C21"
# acceptor = "CC(C)(C(/C=C/c1ccc(c2sc(c3c4cc(c5sc(c6ccc(/C=C/C7C(C#N)/C(OC7(C)C)=C(C#N)\C#N)c7c6nn(CCCCCCCC)n7)cc5C5(c6ccc(CCCCCC)cc6)c6ccc(CCCCCC)cc6)c5c3)c(C4(c3ccc(CCCCCC)cc3)c3ccc(CCCCCC)cc3)c2)c2nn(CCCCCCCC)nc21)C/1C#N)OC1=C(C#N)\C#N"
# acceptor = "O=C(c1cc(F)c(F)cc1C/1=C(C#N)/C#N)C1=C/c1sc(c2cc(C(SCC(CCCCCC)CCCC)(SCC(CCCCCC)CCCC)C3=C4C(SCC(CCCCCC)CCCC)(SCC(CCCCCC)CCCC)c5c3cc(Cc3cc(/C=C6C(c7cc(F)c(F)cc7C/6=C(C#N)\C#N)=O)sc33)c3c5)c4cc2C2)c2c1"
acceptor = "*c1ccc(C2(c3ccc(*)cc3)c3c(sc4cc(/C=C5\C(=O)c6ccccc6C5=C(C#N)C#N)sc34)-c3sc4c5c(sc4c32)-c2sc3cc(/C=C4\C(=O)c6ccccc6C4=C(C#N)C#N)sc3c2C5(c2ccc(*)cc2)c2ccc(*)cc2)cc1"
acceptor_mol = Chem.MolFromSmiles(acceptor)
# add indices to molecule
# for atom in acceptor_mol.GetAtoms():
#     atom.SetAtomMapNum(atom.GetIdx())
Draw.MolToFile(
    acceptor_mol, size=(800, 800), filename=IMAGE_PATH + "test_acceptor_mol.png",
)
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
                        all_dummy_idx.append((idx0, idx1))  # atom index of broken bonds
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
                        all_dummy_idx.append((idx0, idx1))  # atom index of broken bonds
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
                all_dummy_idx.append((idx0, idx1))  # atom index of broken bonds
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
        if atom0.GetDegree() in degree_list and atom1.GetDegree() in degree_list:
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
                all_dummy_idx.append((idx0, idx1))  # atom index of broken bonds
                # set original index property for atoms from broken bonds
                if len(list(atom0.GetPropNames())) != 0:
                    atom0.SetUnsignedProp("og_idx1", idx0)
                else:
                    atom0.SetUnsignedProp("og_idx", idx0)
                if len(list(atom1.GetPropNames())) != 0:
                    atom1.SetUnsignedProp("og_idx1", idx1)
                else:
                    atom1.SetUnsignedProp("og_idx", idx1)

frag_mol = rdmolops.FragmentOnBonds(acceptor_mol, bond_idx_list, addDummies=True)
frag_smiles = Chem.MolToSmiles(frag_mol)
frag_list = frag_smiles.split(".")

Draw.MolToFile(
    frag_mol, size=(800, 800), filename=IMAGE_PATH + "test_acceptor_frag_mol.png",
)

print("all_dummy", all_dummy_idx)

func_grp = []
backbone_grp = []
core_grp = []
core_func_grp = []
core_func_grp_edited = []
core_func_atom_idx = []
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
                    core_func_atom_idx.append(atom.GetUnsignedProp("og_idx"))
                else:
                    n_dummy += 1
    if n_dummy == 1:
        # func_grp.append(Chem.MolToSmiles(mol))
        func_grp.append(mol)
    elif n_dummy == 2 and core_func_bool:
        # core_func_grp.append(Chem.MolToSmiles(mol))
        core_func_grp.append(mol)
    elif n_dummy == 2 and not core_func_bool:
        # backbone_grp.append(Chem.MolToSmiles(mol))
        backbone_grp.append(mol)
    elif n_dummy == 3:
        # backbone_grp.append(Chem.MolToSmiles(mol))
        backbone_grp.append(mol)
    elif n_dummy == 4:
        # core_grp.append(Chem.MolToSmiles(mol))
        core_grp.append(mol)

print("FUNCTIONAL: ", func_grp)
print("BACKBONE: ", backbone_grp)
print("CORE: ", core_grp)
print("CORE FUNCTIONAL: ", core_func_grp)
print("CORE FUNCTIONAL IDX: ", core_func_atom_idx)

core_and_backbone_grp = copy.copy(backbone_grp)
core_and_backbone_grp.extend(core_grp)

# combining core functional fragment (5/6-membered aromatic heterocycle is broken)
# NOTE: keep broken bond index on re-created heterocycle
# add atom with bond index (google RDKit)
# NOTE: fragment connecting only works if core_func_grp is connected to *two* frags
# NOTE: cannot approach from step-by-step building block
# NEW APPROACH: copy adjacent frags, modify them (cut off extra parts)
# keep bonds/atoms in between relevant bonds, add bonds among *three* fragments

# Create aromatic bond for "AddBond" functionality
# cannot instantiate bond class in Python
arene = Chem.MolFromSmiles("c1ccccc1")
for ar_bond in arene.GetBonds():
    aromatic_bond_type = ar_bond.GetBondType()
    break

for mol in core_func_grp:
    frag_to_combine = []
    combined_mol = mol
    for atom in mol.GetAtoms():
        if (
            len(list(atom.GetPropNames())) != 0
            and list(atom.GetPropNames())[0] != "dummyLabel"
        ):
            bond_to_cycle_idx = atom.GetUnsignedProp("og_idx")
            # find matching idx from all_dummy_idx
            result_list_tuples = [
                idx for idx in all_dummy_idx if bond_to_cycle_idx in idx
            ]
            result_core_func = [item for tuple in result_list_tuples for item in tuple]
            result_core_func = [i for i in result_core_func if i != bond_to_cycle_idx]
            print(result_core_func, bond_to_cycle_idx)
    mol_index = 0
    for mol_cycle in core_and_backbone_grp:  # NOTE: needs better name than mol_cycle
        bonded_mol = False
        for atom in mol_cycle.GetAtoms():
            if len(list(atom.GetPropNames())) != 0:
                if atom.GetUnsignedProp("og_idx") in result_core_func:
                    bond_to_cycle_idx = atom.GetUnsignedProp("og_idx")
                    bonded_mol = True  # fragment that was connected to core_func_grp
        if bonded_mol:
            frag_to_combine.append(mol_cycle)
            mol_index += 1

    # generate new heterocycles with correct bonding idx
    # cut frags leaving only relevant bonds/atoms
    for frag in frag_to_combine:
        # NOTE: can delete
        # fragment_atom_idx = []
        # fragment_bond_idx = []
        for atom in frag.GetAtoms():
            if len(list(atom.GetPropNames())) != 0:
                bond_to_cycle_idx = atom.GetUnsignedProp("og_idx")
        # NOTE: can delete
        #         fragment_atom_idx.append(bond_to_cycle_idx)
        #         # find matching idx from all_dummy_idx
        #         result_list_tuples = [
        #             idx for idx in all_dummy_idx if bond_to_cycle_idx in idx
        #         ]
        #         result_core_backbone = [
        #             item for tuple in result_list_tuples for item in tuple
        #         ]
        #         result_core_backbone = [
        #             i for i in result_core_backbone if i != bond_to_cycle_idx
        #         ]
        #         fragment_bond_idx.extend(result_core_backbone)
        # print(fragment_atom_idx, fragment_bond_idx)
        # implement rules for fragmenting important bond only
        # find shortest path between dummy bonds
        frag_atom_idx = []
        for atom in frag.GetAtoms():
            if len(list(atom.GetPropNames())) != 0:
                frag_atom_idx.append(atom.GetIdx())
        combinations = list(itertools.combinations(frag_atom_idx, 2))
        paths = [rdmolops.GetShortestPath(frag, combinations[0][0], combinations[0][1])]
        for combination in combinations:
            if len(
                rdmolops.GetShortestPath(frag, combination[0], combination[1])
            ) == len(paths[0]):
                paths.append(
                    rdmolops.GetShortestPath(frag, combination[0], combination[1])
                )
            elif len(
                rdmolops.GetShortestPath(frag, combination[0], combination[1])
            ) < len(paths[0]):
                paths = [rdmolops.GetShortestPath(frag, combination[0], combination[1])]
        for path in paths:
            relevant = False
            for idx in path:
                relevant_atom = frag.GetAtomWithIdx(idx)
                if len(list(relevant_atom.GetPropNames())) != 0:
                    relevant_atom_prop = relevant_atom.GetUnsignedProp("og_idx")
                    # path contains correct bond index to core_func_grp
                    if relevant_atom_prop in result_core_func:
                        relevant = True
            if relevant:
                best_path = path
        # create fragment with path from frag
        # get bond idx of fragmentation
        irrelevant_bond_idx = []
        for atom_idx in best_path:
            relevant_atom_frag = frag.GetAtomWithIdx(atom_idx)
            for bond in relevant_atom_frag.GetBonds():
                atom0 = bond.GetBeginAtom()
                atom1 = bond.GetEndAtom()
                atom0_idx = bond.GetBeginAtomIdx()
                atom1_idx = bond.GetEndAtomIdx()
                if atom0.GetAtomicNum() == 0 or atom1.GetAtomicNum() == 0:
                    pass
                elif atom0_idx in best_path and atom1_idx in best_path:
                    pass
                else:
                    irrelevant_bond_idx.append(bond.GetIdx())
        # fragment
        frag_of_frag = rdmolops.FragmentOnBonds(
            frag, irrelevant_bond_idx, addDummies=False
        )
        # Draw.MolToFile(
        #     frag_of_frag,
        #     size=(800, 800),
        #     filename=IMAGE_PATH + "test_acceptor_frag_of_frag.png",
        # )
        list_frag_of_frag = list(
            rdmolops.GetMolFrags(frag_of_frag, asMols=True, sanitizeFrags=False)
        )
        # filter for relevant fragment
        for broken_frag in list_frag_of_frag:
            for frag_atom in broken_frag.GetAtoms():
                if len(list(frag_atom.GetPropNames())) != 0:
                    if frag_atom.GetUnsignedProp("og_idx") in result_core_func:
                        relevant_frag = broken_frag
        # Draw.MolToFile(
        #     relevant_frag,
        #     size=(800, 800),
        #     filename=IMAGE_PATH + "test_acceptor_relevant_frag_of_frag.png",
        # )

        # combine frags and core_func_grp
        combined_mol = rdmolops.CombineMols(combined_mol, relevant_frag)

    rw_mol = Chem.RWMol(combined_mol)
    # connect fragments together with proper indices
    for rw_atom in rw_mol.GetAtoms():
        if (
            len(list(atom.GetPropNames())) != 0
            and list(atom.GetPropNames())[0] != "dummyLabel"
        ):
            og_idx = rw_atom.GetUnsignedProp("og_idx")
            # find matching idx from all_dummy_idx
            result_match_tuples = [idx for idx in all_dummy_idx if og_idx in idx]
            result_match = [item for tuple in result_match_tuples for item in tuple]
            result_match = [i for i in result_match if i != og_idx]
            for result_idx in result_match:
                for rw_atom_2 in rw_mol.GetAtoms():
                    if len(list(rw_atom_2.GetPropNames())) != 0:
                        other_og_idx = rw_atom_2.GetUnsignedProp("og_idx")
                        if other_og_idx == result_idx:
                            new_bond_idx0 = rw_atom.GetIdx()
                            new_bond_idx1 = rw_atom_2.GetIdx()
                            if (
                                rw_mol.GetBondBetweenAtoms(new_bond_idx0, new_bond_idx1)
                                == None
                            ):
                                rw_mol.AddBond(
                                    new_bond_idx0,
                                    new_bond_idx1,
                                    order=aromatic_bond_type,
                                )

    # remove all dummy atoms
    rw_mol.BeginBatchEdit()
    for rw_atom_dummy in rw_mol.GetAtoms():
        if rw_atom_dummy.GetAtomicNum() == 0:
            rw_mol.RemoveAtom(rw_atom_dummy.GetIdx())

    rw_mol.CommitBatchEdit()

    frag_result = rw_mol.GetMol()

    # Heterocyclic aromatic 5-membered C ring does have aromatic bonds
    core_func_grp_edited.append(frag_result)
    Draw.MolToFile(
        frag_result,
        size=(800, 800),
        filename=IMAGE_PATH + "test_acceptor_frag_edited.png",
    )

print("CORE FUNCTIONAL GROUP EDITED: ", core_func_grp_edited)

# for visual
# combine all relevant groups
combined_relevant_mol = rdmolops.CombineMols(
    core_func_grp_edited[0], core_func_grp_edited[1]
)
for grp in func_grp:
    rw_grp = Chem.RWMol(grp)
    rw_grp.BeginBatchEdit()
    for rw_atom_dummy in rw_grp.GetAtoms():
        if rw_atom_dummy.GetAtomicNum() == 0:
            rw_grp.RemoveAtom(rw_atom_dummy.GetIdx())

    rw_grp.CommitBatchEdit()
    grp = rw_grp.GetMol()
    combined_relevant_mol = rdmolops.CombineMols(combined_relevant_mol, grp)
for grp in backbone_grp:
    rw_grp = Chem.RWMol(grp)
    rw_grp.BeginBatchEdit()
    for rw_atom_dummy in rw_grp.GetAtoms():
        if rw_atom_dummy.GetAtomicNum() == 0:
            rw_grp.RemoveAtom(rw_atom_dummy.GetIdx())

    rw_grp.CommitBatchEdit()
    grp = rw_grp.GetMol()
    combined_relevant_mol = rdmolops.CombineMols(combined_relevant_mol, grp)
for grp in core_grp:
    rw_grp = Chem.RWMol(grp)
    rw_grp.BeginBatchEdit()
    for rw_atom_dummy in rw_grp.GetAtoms():
        if rw_atom_dummy.GetAtomicNum() == 0:
            rw_grp.RemoveAtom(rw_atom_dummy.GetIdx())

    rw_grp.CommitBatchEdit()
    grp = rw_grp.GetMol()
    combined_relevant_mol = rdmolops.CombineMols(combined_relevant_mol, grp)

Draw.MolToFile(
    combined_relevant_mol,
    size=(800, 800),
    filename=IMAGE_PATH + "test_acceptor_frag_edited.png",
)

