import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import rc

import sys
sys.path.append("../training")
# from code_.training import pipeline_utils
from code_.training.filter_data import filter_dataset, get_appropriate_dataset, get_feature_ids

rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"], "size": 16})

# Data for top panel
with open('../../datasets/Min_2020_n558/cleaned_dataset_mordred.pkl', 'rb') as f:
    mord = pickle.load(f)

data = pd.read_csv('../../datasets/Min_2020_n558/cleaned_dataset.csv')

missing_feats = ['Donor PDI', 'Donor Mn (kDa)', 'Donor Mw (kDa)',
       'D:A ratio (m/m)', 'solvent', 'Active layer spin coating speed (rpm)',
       'total solids conc. (mg/mL)', 'solvent additive',
       'solvent additive conc. (% v/v)', 'active layer thickness (nm)',
       'temperature of thermal annealing', 'annealing time (min)', 'HTL', 'HTL thickness (nm)', 'ETL',
       'ETL thickness (nm)',
       'log hole mobility blend (cm^2 V^-1 s^-1)',
       'log electron mobility blend (cm^2 V^-1 s^-1)',
       'log hole:electron mobility ratio']

num_missing_feats = [data[feat].isnull().sum() for feat in data[missing_feats]]
pct_missing_feats = [100*(feat/len(data)) for feat in num_missing_feats]

missing = pd.DataFrame(list(zip(missing_feats, pct_missing_feats, num_missing_feats)), columns=['feats', '% missing', 'num missing'])

# Data for bottom panel
dataset: pd.DataFrame = get_appropriate_dataset("")

filters: list[str] = ["material properties", "fabrication only", "fabrication only all",
                      "fabrication", "fabrication all",
                      "device architecture", "device architecture all",
                      "log mobilities", "log mobilities all"]

dataset_length: int = len(dataset)

data_subset_size: list[int] = []
for filt in filters:
    feat_ids = get_feature_ids(filt)
    filtered_dataset, _, _ = filter_dataset(dataset, [], feat_ids, [])
    subset_length = len(filtered_dataset)
    data_subset_size.append(subset_length)

data_subset_pct: list[float] = [100 * size / dataset_length for size in data_subset_size]

subsets: pd.DataFrame = pd.DataFrame(list(zip(filters, data_subset_size, data_subset_pct)),
                                     columns=["filters", "size", "percent"])
subsets.at[0, "filters"] = "full dataset"

# Creating a 1x2 layout for subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14))

# Top panel: missing data points
ax1.bar(missing['feats'], missing['% missing'], color='grey', label='Percent')
ax1.set_xlabel('Features')
ax1.set_ylabel('Missing data (%)')
ax1.set_title('Incomplete Features')

# Creating a second y-axis for 'num missing'
ax1r = ax1.twinx()
ax1r.bar(missing['feats'], missing['num missing'], color='grey', label='Num')
ax1r.set_ylabel('Data points')
ax1r.tick_params(axis='y')

# Bottom panel: data subset sizes
ax2.bar(subsets['filters'], subsets['percent'], color='grey', label='Percent')
ax2.set_xlabel('Subsets')
ax2.set_ylabel('Relative dataset size (%)')
ax2.set_title('Relative Subset Sizes')
ax2.tick_params(axis='y')

# Creating a second y-axis for 'size'
ax2r = ax2.twinx()
ax2r.bar(subsets['filters'], subsets['size'], color='grey', label='Size')
ax2r.set_ylabel('Data points')
ax2r.tick_params(axis='y')

# Customizing the top panel
for tick in ax1.get_xticklabels():
    tick.set_rotation(45)
    tick.set_ha('right')

# Customizing the bottom panel
for tick in ax2.get_xticklabels():
    tick.set_rotation(30)
    tick.set_ha('right')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig("missing data and relative subset sizes.png", dpi=300)
# Displaying the combined figure
plt.show()
