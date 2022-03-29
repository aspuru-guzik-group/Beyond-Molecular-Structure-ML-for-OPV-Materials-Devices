from tokenize import group
import seaborn as sns
import pkg_resources
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OPV_ANALYSIS = pkg_resources.resource_filename(
    "opv_ml", "visualization/opv_analysis_master.csv"
)

opv_results = pd.read_csv(OPV_ANALYSIS)

results_df = pd.DataFrame(
    columns=["Model", "Representation", "result_type", "Correlation Coefficient (R)"]
)
results_df["Model"] = ""
results_df["Representation"] = ""
results_df["result_type"] = ""
results_df["Correlation Coefficient (R)"] = ""


representations = ""
results_idx = 0
for index, row in opv_results.iterrows():
    model = opv_results.at[index, "Model"]
    if index == 0:
        representations = row
    elif index >= 1:
        column_count = 0
        result_type_count = 0
        for column in row:
            if column_count > 0:
                column_rep = representations[column_count]
                data = row[column_count]
                results_df.at[results_idx, "Model"] = model
                results_df.at[results_idx, "Representation"] = column_rep
                results_df.at[results_idx, "Correlation Coefficient (R)"] = float(data)
                if result_type_count == 0:
                    results_df.at[results_idx, "result_type"] = "R"
                elif result_type_count == 1:
                    results_df.at[results_idx, "result_type"] = "R_std"
                elif result_type_count == 2:
                    results_df.at[results_idx, "result_type"] = "RMSE"
                elif result_type_count == 3:
                    results_df.at[results_idx, "result_type"] = "RMSE_std"
                    result_type_count = -1
                results_idx += 1
                result_type_count += 1
            column_count += 1

print(results_df)

# create SMILES vs. AugSMILES dataframe
smi_df = results_df[results_df.Representation == "SMILES"]
aug_smi_df = results_df[results_df.Representation == "Augmented SMILES"]

results_smi_df = pd.concat([smi_df, aug_smi_df])
results_smi_R_df = results_smi_df[results_smi_df.result_type == "R"]
results_smi_R_std_df = results_smi_df[results_smi_df.result_type == "R_std"]

# create Manual Frag vs. Augmented Manual Frag dataframe
man_df = results_df[results_df.Representation == "Manual Fragments"]
aug_man_df = results_df[results_df.Representation == "Augmented Manual Fragments"]

results_man_df = pd.concat([man_df, aug_man_df])
results_man_R_df = results_man_df[results_man_df.result_type == "R"]
results_man_R_std_df = results_man_df[results_man_df.result_type == "R_std"]
results_man_R_std_df = results_man_R_std_df.rename(
    columns={"Correlation Coefficient (R)": "R_std"}
)

new_man_df = pd.DataFrame(
    columns=["Model", "Representation", "Correlation Coefficient (R)"]
)
new_man_df["Model"] = results_man_R_df["Model"]
new_man_df["Representation"] = results_man_R_df["Representation"]
new_man_df["Correlation Coefficient (R)"] = results_man_R_df[
    "Correlation Coefficient (R)"
]

# reset index of both dataframes
new_man_df = new_man_df.reset_index()
results_man_R_std_df = results_man_R_std_df.reset_index()

new_man_df = new_man_df.join(results_man_R_std_df["R_std"])


def grouped_barplot(df, cat, subcat, val, err):
    u = df[cat].unique()
    x = np.arange(len(u))
    subx = df[subcat].unique()
    offsets = (np.arange(len(subx)) - np.arange(len(subx)).mean()) / (len(subx) + 1.0)
    width = np.diff(offsets).mean()
    for i, gr in enumerate(subx):
        print(i, gr)
        dfg = df[df[subcat] == gr]
        if i == 0:
            color = (255 / 255, 227 / 255, 179 / 255, 1.0)
        elif i == 1:
            color = (0 / 255, 150 / 255, 82 / 255, 1.0)
        plt.bar(
            x + offsets[i],
            dfg[val].values,
            width=width,
            label="{}".format(gr),
            yerr=dfg[err].values,
            color=color,
            capsize=5,
        )
    plt.xlabel(cat, fontsize=18)
    plt.ylabel(val, fontsize=18)
    plt.ylim(top=0.8)
    plt.xticks(x, u, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.show()


grouped_barplot(
    new_man_df, "Model", "Representation", "Correlation Coefficient (R)", "R_std"
)

