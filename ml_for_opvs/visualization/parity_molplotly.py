import molplotly
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

fab_no_B_smi_data: Path = (
    Path(__file__).parent.parent
    / "data"
    / "input_representation"
    / "OPV_Min"
    / "smiles"
    / "processed_smiles_fabrication_wo_solid"
    / "KFold"
    / "input_test_0.csv"
)

prediction_data: Path = (
    Path(__file__).parent.parent
    / "training"
    / "OPV_Min"
    / "fingerprint"
    / "result_fabrication_wo_solid"
    / "RF_ensemble"
    / "fabrication_wo_solid,solvent_properties"
    / "calc_PCE_percent,FF_percent,Jsc_mA_cm_pow_neg2,Voc_V"
    / "prediction_0.csv"
)


def parity_molplotly(label_data: Path, prediction_data: Path):
    """Plot parity plot with molplotly.
    Args:
        smi_data (Path): _description_
        prediction_data (Path): _description_
    """
    label_data: pd.DataFrame = pd.read_csv(label_data)
    prediction_data: pd.DataFrame = pd.read_csv(prediction_data)
    smi_data: pd.Series = label_data["DA_SMILES"]
    donor_label: pd.Series = label_data["Donor"]
    acceptor_label: pd.Series = label_data["Acceptor"]
    label_data["DA_Label"] = donor_label + "/" + acceptor_label
    prediction_data["smiles"] = smi_data
    prediction_data["DA_Label"] = label_data["DA_Label"]
    print(prediction_data.head())

    # generate a scatter plot
    fig = px.scatter(
        prediction_data,
        x="calc_PCE_percent,FF_percent,Jsc_mA_cm_pow_neg2,Voc_V",
        y="predicted_calc_PCE_percent,FF_percent,Jsc_mA_cm_pow_neg2,Voc_V",
    )
    fig.update_layout(shapes=[
    dict(
      type= 'line',
      yref= 'paper', y0= 0, y1= 1,
      xref= 'x', x0= 5, x1= 5
    )
])
    # add molecules to the plotly graph - returns a Dash app
    app = molplotly.add_molecules(
        fig=fig,
        df=prediction_data,
        smiles_col="smiles",
        title_col="DA_Label",
    )

    # run Dash app inline in notebook (or in an external server)
    app.run_server(mode="inline", port=8700, height=1000)


parity_molplotly(fab_no_B_smi_data, prediction_data)
