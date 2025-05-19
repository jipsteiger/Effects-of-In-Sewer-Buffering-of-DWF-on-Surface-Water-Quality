import pandas as pd
from emprical_sewer_wq import EmpericalSewerWQ
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio
from data.concentration_curves import concentration_dict_ES, concentration_dict_RZ

"""
This file compares the copied Sewer quality model to the one made in WEST. 
It compares both the individual outflows of both the RZ and ES model, 
aswell as the total combined flow to the sewer.
"""

df_west = pd.read_csv(
    rf"data\WEST\WEST_modelRepository\Model_Dommel_Full\comparison.out.txt",
    delimiter="\t",
    header=0,
    index_col=0,
    low_memory=False,
).iloc[1:, :]
start_date = pd.Timestamp("2024-01-01")
df_west["timestamp"] = (
    start_date + pd.to_timedelta(df_west.index.astype(float), unit="D")
).astype("O")
df_west.set_index("timestamp", inplace=True)
df_west = df_west.loc["2024-04-15":"2024-10-15", :]

times = df_west.index
FDs_ES = df_west[".ES_out.FD"].values.astype(float)
H2O_inflows_ES = df_west[".ES_out.Inflow(H2O_sew)"].values.astype(float)
FDs_RZ = df_west[".RZ_out.FD"].values.astype(float)
H2O_inflows_RZ = df_west[".RZ_out.Inflow(H2O_sew)"].values.astype(float)

WQ_ES = EmpericalSewerWQ(
    concentration_dict=concentration_dict_ES,
    COD_av=546,
    CODs_av=158,
    NH4_av=44,
    PO4_av=7.1,
    TSS_av=255,
    Q_95_av=70800,
    alpha_CODs=0.8,
    alpha_NH4=0.8,
    alpha_PO4=0.8,
    alpha_TSS=0.8,
    beta_CODs=0.8,
    beta_NH4=0.8,
    beta_PO4=0.8,
    beta_TSS=0.8,
)
WQ_RZ = EmpericalSewerWQ(
    concentration_dict=concentration_dict_RZ,
    COD_av=573,
    CODs_av=206,
    NH4_av=44,
    PO4_av=7.1,
    TSS_av=203,
    Q_95_av=58800,
    alpha_CODs=0.8,
    alpha_NH4=0.8,
    alpha_PO4=0.8,
    alpha_TSS=0.8,
    beta_CODs=0.8,
    beta_NH4=0.8,
    beta_PO4=0.8,
    beta_TSS=0.8,
    proc4_slope1_CODs=0.288,
    proc4_slope1_NH4=0.288,
    proc4_slope1_PO4=0.288,
    proc4_slope2_CODs=0.576,
    proc4_slope2_NH4=0.576,
    proc4_slope2_PO4=0.576,
    Q_proc6=120000,
    proc6_slope2_COD=864,
    proc6_slope2_TSS=864,
    proc6_t1_COD=3,
    proc6_t1_TSS=3,
    proc6_t2_COD=12,
    proc6_t2_TSS=12,
    proc7_slope1_COD=23040,
    proc7_slope1_TSS=23040,
    proc7_slope2_COD=5760,
    proc4_slope2_TSS=5760,
)

for time, FD_ES, H2O_inflow_ES, FD_RZ, H2O_inflow_RZ in zip(
    times, FDs_ES, H2O_inflows_ES, FDs_RZ, H2O_inflows_RZ
):
    WQ_ES.update(time, H2O_inflow_ES / 1_000_000, FD_ES)
    WQ_RZ.update(time, H2O_inflow_RZ / 1_000_000, FD_RZ)
WQ_ES.write_output_log("compare_ES")
WQ_RZ.write_output_log("compare_RZ")


fig = go.Figure()
for key in df_west.keys():
    if "_out.Outflow" in key:
        fig.add_trace(
            go.Scatter(
                x=df_west.index,
                y=abs(df_west[key].astype(float)),
                mode="lines",
                name=f"West {key}",
            )
        )

model_result_ES = pd.read_csv("output_effluent/compare_ES.Effluent.csv", index_col=0)
model_result_RZ = pd.read_csv("output_effluent/compare_RZ.Effluent.csv", index_col=0)

for key in model_result_ES.keys():
    fig.add_trace(
        go.Scatter(
            x=df_west.index,
            y=abs(model_result_ES[key].astype(float)),
            mode="lines",
            name=f"Jip Model ES {key}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_west.index,
            y=abs(model_result_RZ[key].astype(float)),
            mode="lines",
            name=f"Jip Model RZ {key}",
        )
    )

pio.show(fig, renderer="browser")

fig = go.Figure()
for key in df_west.keys():
    if "Well_35" in key:
        fig.add_trace(
            go.Scatter(
                x=df_west.index,
                y=df_west[key].astype(float),
                mode="lines",
                name=f"West combined{key}",
            )
        )
        pollutant = key.split("(")[-1].split(")")[0]
        NS = df_west[f".NS_out.Outflow({pollutant})"].values.astype(float)
        ES = model_result_ES.loc[:, pollutant].values
        RZ = model_result_RZ.loc[:, pollutant].values
        combined = NS + ES + RZ
        fig.add_trace(
            go.Scatter(
                x=df_west.index,
                y=combined,
                mode="lines",
                name=f"SWMM combined {key}",
            )
        )
pio.show(fig, renderer="browser")
