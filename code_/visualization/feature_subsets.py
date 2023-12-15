import json

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

# Assuming you have loaded your JSON data into these variables: first_json, second_json
subsets_file = Path('../training/subsets.json')
with subsets_file.open('r') as f:
    subsets_json = json.load(f)
filters_file = Path('../training/filters.json')
with filters_file.open('r') as f:
    filters_json = json.load(f)

# Extracting filter names and subset names
filters = ['material properties', 'fabrication only', 'fabrication only all', 'fabrication', 'fabrication all', 'device architecture', 'device architecture all', 'log mobilities', 'log mobilities all']
subsets = ['material labels', 'material properties', 'material properties missing', 'fabrication labels', 'fabrication', 'fabrication missing', 'device architecture labels', 'device architecture', 'device architecture missing', 'mobilities', 'log mobilities']

# Get all relevant column names from subsets
columns = ['Donor', 'Acceptor', 'HOMO_D (eV)', 'LUMO_D (eV)', 'Eg_D (eV)', 'Ehl_D (eV)', 'HOMO_A (eV)', 'LUMO_A (eV)', 'Eg_A (eV)', 'Ehl_A (eV)', 'Donor PDI', 'Donor Mn (kDa)', 'Donor Mw (kDa)', 'solvent', 'solvent additive', 'D:A ratio (m/m)', 'solvent additive conc. (% v/v)', 'temperature of thermal annealing', 'Active layer spin coating speed (rpm)', 'total solids conc. (mg/mL)', 'annealing time (min)', 'HTL energy level (eV)', 'ETL energy level (eV)', 'active layer thickness (nm)', 'HTL thickness (nm)', 'ETL thickness (nm)', 'log hole mobility blend (cm^2 V^-1 s^-1)', 'log electron mobility blend (cm^2 V^-1 s^-1)', 'log hole:electron mobility ratio']

# Creating a matrix to represent presence of column names in filters
data_matrix = np.zeros((len(columns), len(filters)))

for idx_filter, filter_name in enumerate(filters):
    for idx_column, column_name in enumerate(columns):
        # Check if column is present in the filter's subsets
        for subset_name in filters_json.get(filter_name, []):
            if column_name in subsets_json.get(subset_name, []):
                data_matrix[idx_column, idx_filter] = 1  # Column is in the filter

# Manually adding 'Donor' and 'Acceptor' since they were handled differently
for idx_filter in [0, 3, 4, 5, 6, 7, 8]:
    data_matrix[columns.index('Donor'), idx_filter] = 1
    data_matrix[columns.index('Acceptor'), idx_filter] = 1

# Manually adding in 'solvent' and 'solvent additive' since they were handled differently
for idx_filter in range(1, 9):
    data_matrix[columns.index('solvent'), idx_filter] = 1
    data_matrix[columns.index('solvent additive'), idx_filter] = 1


# Creating the plot
plt.figure(figsize=(9, 11))
plt.imshow(data_matrix, cmap='Greens', aspect='auto')
# plt.matshow(data_matrix, cmap='Greens')

# Set ticks and labels
plt.xticks(np.arange(len(filters)), filters, rotation=45, ha='right')
plt.yticks(np.arange(len(columns)), columns)
plt.xlabel('Filters')
plt.ylabel('Features')
plt.title('Presence of Features in Filters')


# plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)
# plt.gca().set_xticks(np.arange(len(filters) + 1) - 0.5, minor=False)
# plt.gca().set_yticks(np.arange(len(columns) + 1) - 0.5, minor=False)

# Show plot
plt.tight_layout()
plt.savefig('feature_subsets.png', dpi=600)
plt.show()

