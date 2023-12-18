import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path

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

pct_missing_feats = [100*(data[feat].isnull().sum()/len(data)) for feat in data[missing_feats]]

missing = pd.DataFrame(list(zip(missing_feats, pct_missing_feats)), columns=['feats', '% missing'])

sns.barplot(missing, x='feats', y='% missing', color='grey')

# # Rotate x-axis labels
# for tick_label in ax.get_xticklabels():
#     tick_label.set_rotation(45)
plt.xticks(np.arange(len(missing['feats'])), missing['feats'], rotation=45, ha='right')

plt.tight_layout()
plt.savefig("percent missing from features.png", dpi=300)
plt.show()
