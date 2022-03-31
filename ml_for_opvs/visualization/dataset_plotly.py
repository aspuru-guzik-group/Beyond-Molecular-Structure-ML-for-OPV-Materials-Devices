import plotly.express as px
import pkg_resources
import pandas as pd

DATASET_ANALYSIS = pkg_resources.resource_filename(
    "ml_for_opvs", "visualization/dataset_size.csv"
)


data_results = pd.read_csv(DATASET_ANALYSIS)
print(data_results)

fig = px.bar(
    data_results,
    x="Dataset Type",
    y="Number of Data Points",
    title="Dataset Size Comparison",
)
y_text = data_results["Number of Data Points"]
fig.update_traces(
    text=y_text, texttemplate="%{text}", textfont_family="Roboto", marker_color="green",
)
fig.update_layout(font_family="Roboto")
fig.update_xaxes(side="bottom")
fig.layout.height = 600
fig.layout.width = 1100
fig.show()
