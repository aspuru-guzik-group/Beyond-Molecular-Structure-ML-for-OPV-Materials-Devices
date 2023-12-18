import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# from ..training.filter_data import filter_dataset, get_appropriate_dataset, get_feature_ids
import sys
sys.path.append("../training")
# from training.filter_data import filter_dataset, get_appropriate_dataset, get_feature_ids
from code_.training import pipeline_utils
from code_.training.filter_data import filter_dataset, get_appropriate_dataset, get_feature_ids

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


# data_subset_size: list[int] = [len(filter_dataset(dataset, [], get_feature_ids(filt), [])) for filt in filters]
data_subset_pct: list[float] = [100 * size / dataset_length for size in data_subset_size]

subsets: pd.DataFrame = pd.DataFrame(list(zip(filters, data_subset_size, data_subset_pct)),
                                     columns=["filters", "size", "percent"])

subsets.at[0, "filters"] = "full dataset"

# plt.figure(figsize=(8, 6))
fig, ax1 = plt.subplots()
# Plotting the bars for 'percent' on the left y-axis
ax1.bar(subsets['filters'], subsets['percent'], color='grey', label='Percent')
ax1.set_xlabel('Subsets')
ax1.set_ylabel('Relative dataset size (%)')
ax1.tick_params(axis='y')

# Creating a second y-axis for 'size'
ax2 = ax1.twinx()
ax2.bar(subsets['filters'], subsets['size'], color='grey', label='Size')
ax2.set_ylabel('Data points')
ax2.tick_params(axis='y')

# Customizing the plot
# plt.title('Bar Plot with Dual Y-Axis')
# plt.xticks(np.arange(len(subsets['filters'])), subsets['filters'], rotation=45, ha='right')
for tick in ax1.get_xticklabels():
    tick.set_rotation(45)
    tick.set_ha('right')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig("relative subset sizes.png", dpi=300)
plt.show()
