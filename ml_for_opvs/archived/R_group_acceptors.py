import pkg_resources
from rdkit import Chem
from rdkit.Chem.rdmolops import ReplaceSubstructs
import csv

ACC_PATH = pkg_resources.resource_filename("opv_ml", "min_acceptors_smiles.txt")
CSV_PATH = pkg_resources.resource_filename("opv_ml", "min_acceptors_smiles.csv")

# NOTE: Need to access masters file and get corresponding SMILES and Label in a .csv file
# Label | SMILES | SMILES (w/ substituted R) | BigSMILES | SELFIES | PCE


class R_acceptor:
    """
    Class containing functions required to change R group for acceptor structures
    """

    def __init__(self, acceptor_path):
        file = open(acceptor_path, "r")
        self.data = file.read()
        file.close()

    def PreProcess(self):
        """Function that will separate long list of SMILES

        Parameters
        ----------
        None

        Returns
        ----------
        data_list: list of molecules separated by "."
        """
        self.data_list = self.data.split(".")

    def Process(self):
        """Function that will prepare SMILES: 
        - 
        - insert R group
        - account for variable attachment

        Parameters
        ----------
        smiles_list: list of smiles and other accessory

        Returns
        ----------
        smiles_dict: dictionary for name of molecule, SMILES, and molecular object
        """
        # patts dictionary contains all R group to SMILES
        patts = {
            "[R1]": "CC(CCCCCC)CCCCCCCC",
            "[R2]": "CCCCCCCC",
            "[R3]": "[Si](CCC)(CCC)(CCC)",
            "[R4]": "CC(CC)CCCC",
            "[R5]": "SCCCCCCCCCCCC",
            "[R6]": "CC(CCCCCCCC)CCCCCCCCCC",
            "[R7]": "SCC(CCCCCC)CCCC",
            "[R8]": "[Si](CC)(CC)(CC)",
            "[R9]": "[Si](C(C)C)(C(C)C)C(C)C",
            "[R10]": "[Si](CCCC)(CCCC)(CCCC)",
            "[R11]": "[Si](C)(C)CCCCCCCC",
            "[R12]": "SCCCCC=C",
            "[R13]": "SCC4CCCCC4",
            "[R14]": "CCCCCC",
            "[R15]": "CCCCCCCCCC",
            "[R19]": "CCCCCCCCCCCCCCCC",
            "[R20]": "CCCCCCCCCCC",
            "[R21]": "C(CCCCCCCCC)CCCCCCC",
            "[R24]": "COCCOC",
            "[R27]": "CCCC",
        }
        """
        i = 0
        while i < len(self.data_list):
            string = self.data_list[i]
            for r in patts:
                string = string.replace(r, patts[r])
            self.data_list[i] = string
            i += 1
        """

        self.smiles_list = []
        i = 0
        halogen_count = 0
        # NOTE: need to remove duplicates that don't have the halogen attachment
        while i < len(self.data_list):
            smile_string = self.data_list[i]
            if smile_string[0] == "[" and smile_string[1] == "]":
                halogen_count += 1
                if halogen_count == 2:
                    halogen_count = 0
                    halogen = smile_string[2]
                    if halogen == "C":
                        halogen = "Cl"
                    elif halogen == "B":
                        halogen = "Br"
                    previous_smile = self.data_list[i - 2]
                    # print(type(previous_smile))
                    # print("previous:", previous_smile)
                    mol_object = Chem.MolFromSmarts(previous_smile)
                    # print(mol_object)
                    indanone_object = Chem.MolFromSmarts(
                        "C=C8C(c9ccccc9C\8=C(C#N)\C#N)=O"
                    )
                    substitute_smiles = (
                        "C=C8C(c9cc(" + halogen + ")ccc9C\8=C(C#N)\C#N)=O"
                    )
                    substitute_object = Chem.MolFromSmarts(substitute_smiles)
                    new_mol_object = ReplaceSubstructs(
                        mol_object, indanone_object, substitute_object, replaceAll=True,
                    )
                    new_smile = Chem.MolToSmiles(new_mol_object[0])
                    new_smile = Chem.CanonSmiles(new_smile)
                    self.smiles_list.append([new_smile])
            else:
                self.smiles_list.append([smile_string])

            i += 1
        print(i)

    def smiles_to_csv(self, csv_path):
        with open(csv_path, "w", newline="") as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(["SMILE"])
            wr.writerows(self.smiles_list)


acceptor = R_acceptor(ACC_PATH)
acceptor.PreProcess()
acceptor.Process()
acceptor.smiles_to_csv(CSV_PATH)
