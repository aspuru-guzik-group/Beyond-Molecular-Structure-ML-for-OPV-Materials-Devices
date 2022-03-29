from rdkit.Chem import PandasTools
import pandas as pd
import pkg_resources

SDF_PATH = pkg_resources.resource_filename("r_group", "min_donors_without_names.sdf")

frame = PandasTools.LoadSDF(
    SDF_PATH, smilesName="SMILES", molColName="Molecule", includeFingerprints=True
)

print(frame)
