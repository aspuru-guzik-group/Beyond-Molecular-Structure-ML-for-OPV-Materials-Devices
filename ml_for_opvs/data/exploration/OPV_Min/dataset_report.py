import json
import pandas as pd
from typing import Dict, Union
from pathlib import Path

opv_file: Path = (
    Path.home() / "Downloads" / "FINAL Machine Learning OPV Parameters.xlsx"
)
output: Path = Path("__file__").parent.absolute() / "dataset_report"
output.mkdir(parents=True, exist_ok=True)

opv_data: pd.DataFrame = pd.read_excel(opv_file)

# Define subsets
refs: set = {"index", "ref number from paper"}
outputs: set = {"Voc (V)", "Jsc (mA cm^-2)", "FF (%)", "PCE (%)"}
labels_molecules: set = {"Donor Molecule", "Acceptor Molecule"}
# labels_properties: set = {'HOMO_D (eV)', 'LUMO_D (eV)', 'HOMO_A (eV)', 'LUMO_A (eV)' }
labels_fabrication: set = {
    "D:A ratio (m/m)",
    "solvent",
    "total solids conc. (mg/mL)",
    "solvent additive",
    "solvent additive conc. (% v/v)",
    "temperature of thermal annealing",
}
labels_device: set = {
    "active layer thickness (nm) ",
    "hole contact layer",
    "electron contact layer",
}
labels_electrical: set = {
    "hole mobility blend (cm^2 V^-1 s^-1)",
    "electron mobility blend",
}

labels_fabrication_wo_solids: set = {
    "D:A ratio (m/m)",
    "solvent",
    "solvent additive",
    "solvent additive conc. (% v/v) ",
    "temperature of thermal annealing ",
}
labels_device_wo_thickness: set = {"hole contact layer", "electron contact layer"}

filters: Dict[str, set] = {
    "fabrication": (refs | outputs | labels_molecules | labels_fabrication),
    "device": (refs | outputs | labels_molecules | labels_fabrication | labels_device),
    "electrical": (
        refs
        | outputs
        | labels_molecules
        | labels_fabrication
        | labels_device
        | labels_electrical
    ),
    "fabrication_wo_solids": (
        refs | outputs | labels_molecules | labels_fabrication_wo_solids
    ),
    "device_wo_solids_thick": (
        refs
        | outputs
        | labels_molecules
        | labels_fabrication_wo_solids
        | labels_device_wo_thickness
    ),
}

# Get available subsets for ML
available: Dict[str, pd.DataFrame] = {
    "fabrication": opv_data[filters["fabrication"]].dropna(),
    "device": opv_data[filters["device"]].dropna(),
    "electrical": opv_data[filters["electrical"]].dropna(),
    "fabrication_wo_solids": opv_data[filters["fabrication_wo_solids"]].dropna(),
    "device_wo_solids_thick": opv_data[filters["device_wo_solids_thick"]].dropna(),
}
for key in available.keys():
    available[key].to_excel((output / f"available_{key}.xlsx"), index=False)

# Get unique values
unique: Dict[str, set] = {
    "donors": set(opv_data["Donor Molecule"]),
    "acceptors": set(opv_data["Acceptor Molecule"]),
    "solvents": set(opv_data["solvent"]),
    "additives": set(opv_data["solvent additive"]),
    "hole contact layers": set(opv_data["hole contact layer"]),
    "electron contact layers": set(opv_data["electron contact layer"]),
}
for key in unique.keys():
    pd.DataFrame(data=unique[key]).dropna().to_csv(
        (output / f"unique_{key}.csv"), index=False
    )

# Data summary
data_summary: Dict[str, int] = {
    "data points": len(opv_data),
    "donor labels": len(unique["donors"]),
    "acceptor labels": len(unique["acceptors"]),
    "solvent labels": len(unique["solvents"])
    - 1,  # Substracting 1 to account for the nan value
    "additive labels": len(unique["additives"]) - 1,
    "HCL labels": len(unique["hole contact layers"]) - 1,
    "ECL labels": len(unique["electron contact layers"]) - 1,
}
with (output / "data_summary.json").open(mode="w") as f:
    json.dump(data_summary, f)

# Subset sizes + %
subset_summary: Dict[str, Union[int, float]] = {
    "fabrication (raw)": len(available["fabrication"]),
    "fabrication (%)": len(available["fabrication"]) / data_summary["data points"],
    "device (raw)": len(available["device"]),
    "device (%)": len(available["device"]) / data_summary["data points"],
    "electrical (raw)": len(available["electrical"]),
    "electrical (%)": len(available["electrical"]) / data_summary["data points"],
    "fabrication w/o solids (raw)": len(available["fabrication_wo_solids"]),
    "fabrication w/o solids (%)": len(available["fabrication_wo_solids"])
    / data_summary["data points"],
    "device w/o solids or thickness (raw)": len(available["device_wo_solids_thick"]),
    "device w/o solids or thickness (%)": len(available["device_wo_solids_thick"])
    / data_summary["data points"],
}
with (output / "subset_sizes.json").open(mode="w") as f:
    json.dump(subset_summary, f)

# Missing data points
missing_data: pd.DataFrame = pd.read_excel(opv_file, sheet_name="Missing data")
missing_data.to_csv((output / "missing_data.csv"), index=False)
