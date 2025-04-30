import pandas as pd
from emprical_sewer_wq import EmpericalSewerWQ
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio

df_west = pd.read_csv(
    rf"data\WEST\SWMM_inputs_dwf_and_precipitation\concentration_check.out.txt",
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

times = df_west.index
FDs = df_west[".ES_out.FD"].values.astype(float)
H2O_inflows = df_west[".ES_out.Inflow(H2O_sew)"].values.astype(float)

WQ_ES = EmpericalSewerWQ(
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

for time, FD, H2O_inflow in zip(times, FDs, H2O_inflows):
    WQ_ES.update(time, H2O_inflow / 1_000_000, FD)
WQ_ES.write_output("compare_ES")


fig = go.Figure()
for key in df_west.keys():
    if not ".RZ_out" in key:
        fig.add_trace(
            go.Scatter(
                x=df_west.index,
                y=abs(df_west[key].astype(float)),
                mode="lines",
                name=f"West {key}",
            )
        )

model_result = pd.read_csv("compare_ES.Effluent.csv", index_col=0)

for key in model_result.keys():
    fig.add_trace(
        go.Scatter(
            x=df_west.index,
            y=abs(model_result[key].astype(float)),
            mode="lines",
            name=f"Jip Model {key}",
        )
    )

pio.show(fig, renderer="browser")
