import numpy as np
import plotly.express as px
import pkg_resources
import pandas as pd
import copy
import plotly.io as pio

OPV_ANALYSIS = pkg_resources.resource_filename(
    "opv_ml", "visualization/opv_analysis_master.csv"
)


opv_results = pd.read_csv(OPV_ANALYSIS)

data_r = []
data_r_std = []
data_rmse = []
data_rmse_std = []

row_count = 0
for index, row in opv_results.iterrows():
    if row_count >= 1:
        results = row[1:]
        result_idx = 0
        r = []
        r_std = []
        rmse = []
        rmse_std = []
        for i in range(len(results) - 1):
            if result_idx == 0:
                r.append(round(float(results[i]), 3))
            elif result_idx == 1:
                r_std.append(round(float(results[i]), 3))
            elif result_idx == 2:
                rmse.append(round(float(results[i]), 3))
            elif result_idx == 3:
                rmse_std.append(round(float(results[i]), 3))
                result_idx = -1
            result_idx += 1
        data_r.append(r)
        data_r_std.append(r_std)
        data_rmse.append(rmse)
        data_rmse_std.append(rmse_std)
    row_count += 1

# create z_text which has the std error labels in ()
outer_idx = 0
z_text = copy.deepcopy(data_r)
while outer_idx < len(data_r):
    inner_idx = 0
    while inner_idx < len(data_r[outer_idx]):
        z_text[outer_idx][inner_idx] = (
            str(data_r[outer_idx][inner_idx])
            + "<br>"
            + "("
            + str(data_r_std[outer_idx][inner_idx])
            + ")"
        )
        inner_idx += 1
    outer_idx += 1

print(data_r)

fig = px.imshow(
    data_r,
    labels=dict(
        x="Data Representation", y="ML/DL Model", color="Correlation Coefficient (R)"
    ),
    x=[
        "SMILES (n=386)",
        "BigSMILES (n=386)",
        "SELFIES (n=386)",
        "Augmented SMILES (n~9700)",
        "Fragments by Hand-Written Rules (n=386)",
        "Augmented Hand-Written Fragments (n~2000)",
        "Fragments by BRICS (n=386)",
        "Manual Fragments (n=386)",
        "Augmented Manual Fragments (n~2000)",
        "Morgan Fingerprints",
    ],
    y=[
        "Random Forest (RF, cv=5)",
        "Boosted Regression Tree (BRT, cv=5)",
        "Support Vector Regression (SVR, cv=5)",
        "Neural Network (NN, cv=5)",
        "Long-Short-Term Memory (LSTM, cv=5)",
    ],
    text_auto=True,
    color_continuous_scale="RdYlGn",
    aspect="auto",
)
fig.update_traces(
    text=z_text,
    texttemplate="%{text}",
    textfont_size=16,
    textfont_family="Roboto",
    colorbar_tickfont_family="Roboto",
    colorbar_tickfont_size=40,
    colorbar_title_font_family="Roboto",
    colorbar_title_font_size=40,
    legendgrouptitle_font_size=40,
)
fig.update_xaxes(side="top")
fig.layout.height = 600
fig.layout.width = 1700
fig.show()

