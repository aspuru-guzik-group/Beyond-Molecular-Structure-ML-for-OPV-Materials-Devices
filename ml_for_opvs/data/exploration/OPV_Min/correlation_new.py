from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class HeatmapGenerator:
    def __init__(self, df_path):
        self.full_df = pd.read_csv(df_path)
        self.df = self.full_df[["dipole", "dD", "dP", "dH", "dHDon", "dHAcc", "MW", "Density", "BPt", "MPt", "logKow",
                                "RI", "Trouton", "RER", "ParachorGA", "RD", "DCp", "log n", "SurfTen"]
                               ]

    def plot_heatmap(self):
        corr = self.df.corr(method="pearson")
        sns.set(style="white")
        cmap = sns.color_palette("icefire", as_cmap=True)
        f, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corr, cmap=cmap, vmin=-1, vmax=1, center=0,
                    square=True, linewidths=.5, annot=True, fmt=".2f",
                    cbar_kws={"shrink": .5}, annot_kws={"fontsize":10})
        ax.set_title('Pearson Correlation Heatmap')
        plt.show()


if __name__ == "__main__":
    df_path = Path.home() / "Downloads" / "HSPiP solvent properties.csv"
    heatmap_generator = HeatmapGenerator(df_path)
    heatmap_generator.plot_heatmap()
