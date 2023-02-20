import json
import pandas as pd
from typing import Dict, Union
from pathlib import Path

opv_file: Path = (
    Path.home() / "Downloads" / "OPV ML data extraction.xlsx"
)
output: Path = Path("__file__").parent.absolute() / "dataset_report"
output.mkdir(parents=True, exist_ok=True)

opv_data: pd.DataFrame = pd.read_excel(opv_file)

# Define subsets
refs: set = {"ref", "ref number from paper"}
outputs: set = {"Voc (V)", "Jsc (mA cm^-2)", "FF (%)", "PCE (%)"}
labels_molecules: set = {"Donor Molecule", "Acceptor Molecule"}
labels_properties_A: set = {
    'HOMO_D (eV)',
    'LUMO_D (eV)',
    'Eg_D (eV)',
    'HOMO_A (eV)',
    'LUMO_A (eV)',
    'Eg_A (eV)',
}
labels_properties_B: set = {
    'Donor PDI',
    'Donor Mn (kDa)',
    'Donor Mw (kDa)',
}
labels_fabrication_A: set = {
    "D:A ratio (m/m)",
    "solvent",
    "solvent additive",
    "solvent additive conc. (% v/v)",
    "temperature of thermal annealing",
}
labels_fabrication_B: set = {
    'Active layer spin coating speed (rpm)',
    "total solids conc. (mg/mL)",
    'annealing time (min)',
}
labels_device_A: set = {
    "hole contact layer",
    "electron contact layer",
}
labels_device_B: set = {
    "active layer thickness (nm)",
    "HTL thickness (nm)",
    "ETL thickness (nm)",
}
labels_mobility: set = {
    "hole mobility blend (cm^2 V^-1 s^-1)",
    "electron mobility blend (cm^2 V^-1 s^-1)",
}

filters: Dict[str, set] = {
    "fabrication": (refs | outputs | labels_molecules | labels_fabrication_A | labels_fabrication_B),
    "device": (refs | outputs | labels_molecules | labels_fabrication_A | labels_fabrication_B | labels_device_A | labels_device_B),
    "mobility": (
            refs
            | outputs
            | labels_molecules
            | labels_fabrication_A
            | labels_fabrication_B
            | labels_device_A
            | labels_device_B
            | labels_mobility
    ),
    "fabrication_noB": (
        refs | outputs | labels_molecules | labels_fabrication_A
    ),
    "device_noB": (
        refs
        | outputs
        | labels_molecules
        | labels_fabrication_A
        | labels_device_A
    ),
    "mobility_noB": (
            refs
            | outputs
            | labels_molecules
            | labels_fabrication_A
            | labels_device_A
            | labels_mobility
    ),
}

# Get available subsets for ML
available: Dict[str, pd.DataFrame] = {
    "fabrication": opv_data[filters["fabrication"]].dropna(),
    "device": opv_data[filters["device"]].dropna(),
    "mobility": opv_data[filters["mobility"]].dropna(),
    "fabrication_noB": opv_data[filters["fabrication_noB"]].dropna(),
    "device_noB": opv_data[filters["device_noB"]].dropna(),
    "mobility_noB": opv_data[filters["mobility_noB"]].dropna(),
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
labels_summary: Dict[str, int] = {
    "data points": len(opv_data),
    "donor labels": len(unique["donors"]),
    "acceptor labels": len(unique["acceptors"]),
    "solvent labels": len(unique["solvents"]) - 1,  # Substracting 1 to account for the nan value
    "additive labels": len(unique["additives"]) - 1,
    "HCL labels": len(unique["hole contact layers"]) - 1,
    "ECL labels": len(unique["electron contact layers"]) - 1,
}
with (output / "labels_summary.json").open(mode="w") as f:
    json.dump(labels_summary, f)

# Subset sizes + %
subset_summary: Dict[str, Union[int, float]] = {
    "fabrication (raw)": len(available["fabrication"]),
    "fabrication (%)":   len(available["fabrication"]) / labels_summary["data points"],
    "device (raw)": len(available["device"]),
    "device (%)":        len(available["device"]) / labels_summary["data points"],
    "mobility (raw)": len(available["mobility"]),
    "mobility (%)":    len(available["mobility"]) / labels_summary["data points"],
    "fabrication no B (raw)": len(available["fabrication_noB"]),
    "fabrication no B (%)": len(available["fabrication_noB"]) / labels_summary["data points"],
    "device no B (raw)": len(available["device_noB"]),
    "device no B (%)": len(available["device_noB"]) / labels_summary["data points"],
    "mobility no B (raw)": len(available["mobility_noB"]),
    "mobility no B (%)": len(available["mobility_noB"]) / labels_summary["data points"],
}
with (output / "subset_sizes.json").open(mode="w") as f:
    json.dump(subset_summary, f)

# Missing data points
missing_data: pd.DataFrame = pd.read_excel(opv_file, sheet_name="Missing data")
missing_data.to_csv((output / "missing_data.csv"), index=False)
