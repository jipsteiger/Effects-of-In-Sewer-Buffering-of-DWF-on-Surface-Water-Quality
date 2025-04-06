import swmm_api as sa
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
import matplotlib.dates as mdates

model_name = "model_jip_geen_regen"


def make_dwf_only():
    aa = sa.SwmmOutput(rf"data\SWMM\{model_name}.out").to_frame()

    for outfall in ["out_RZ", "out_ES"]:
        timeseries = aa.node[outfall]["total_inflow"]

        dec_index = (timeseries.index - pd.Timestamp("2023-01-01")).total_seconds() / (
            24 * 60 * 60
        )

        H2O_sew = timeseries.values * 24 * 3600  # From CMS to m3/d
        NH4_sew = [0] * len(H2O_sew)
        PO4_sew = [0] * len(H2O_sew)
        COD_sol = [0] * len(H2O_sew)
        X_TSS_sew = [0] * len(H2O_sew)
        COD_part = [0] * len(H2O_sew)
        west_values = {
            "H2O_sew": H2O_sew,
            "NH4_sew": NH4_sew,
            "PO4_sew": PO4_sew,
            "COD_sol": COD_sol,
            "X_TSS_sew": X_TSS_sew,
            "COD_part": COD_part,
        }
        df = pd.DataFrame(west_values, index=dec_index)

        df_csv = pd.DataFrame(west_values, index=timeseries.index)

        output_file = f"#.t\tH2O_sew\tNH4_sew\tPO4_sew\tCOD_sol\tX_TSS_sew\tCOD_part\n#d\tm3/d\tm3/d\tm3/d\tm3/d\tm3/d\tm3/d\n"
        output_file += df[:].to_csv(sep="\t", header=False)

        static_name = "latest"  # Change this as needed
        with open(f"swmm_output/{static_name}_{outfall}_DWF_only_out.txt", "w") as f:
            f.write(output_file)
        df_csv.to_csv(
            f"swmm_output/{static_name}_{outfall}_DWF_only_out.csv",
            sep=";",
            decimal=",",
        )


def make_dwf_only_constant():
    aa = sa.SwmmOutput(rf"data\SWMM\{model_name}.out").to_frame()

    for outfall in ["out_RZ", "out_ES"]:
        timeseries = aa.node[outfall]["total_inflow"]
        value = (timeseries.sum() / len(timeseries)) * (24 * 3600)  # From CMS to m3/d

        dec_index = (timeseries.index - pd.Timestamp("2023-01-01")).total_seconds() / (
            24 * 60 * 60
        )

        H2O_sew = [value] * len(timeseries)
        west_values = {
            "H2O_sew": H2O_sew,
        }
        df = pd.DataFrame(west_values, index=dec_index)

        df_csv = pd.DataFrame(west_values, index=timeseries.index)

        output_file = f"#.t\tH2O_sew\n#d\tm3/d\n"
        output_file += df[:].to_csv(sep="\t", header=False)

        static_name = "latest"  # Change this as needed
        with open(
            f"swmm_output/{static_name}_{outfall}_only_constant_DWF.txt", "w"
        ) as f:
            f.write(output_file)
        df_csv.to_csv(
            f"swmm_output/{static_name}_{outfall}_only_constant_DWF.csv",
            sep=";",
            decimal=",",
        )


def timeseries_snippet():
    es_dwf = pd.read_csv(
        rf"swmm_output\latest_out_ES_DWF_only_out.csv",
        index_col=0,
        parse_dates=True,
        delimiter=";",
        decimal=",",
    )
    es_dwf_const = pd.read_csv(
        rf"swmm_output\latest_out_ES_only_constant_DWF.csv",
        index_col=0,
        parse_dates=True,
        delimiter=";",
        decimal=",",
    )
    es = pd.read_csv(
        rf"swmm_output\latest_out_ES_out_WEST_precipitation.csv",
        index_col=0,
        parse_dates=True,
        delimiter=";",
        decimal=",",
    )
    es_base = pd.read_csv(
        rf"swmm_output\latest_out_ES_out_base.csv",
        index_col=0,
        parse_dates=True,
        delimiter=";",
        decimal=",",
    )
    rz_dwf = pd.read_csv(
        "swmm_output\latest_out_RZ_DWF_only_out.csv",
        index_col=0,
        parse_dates=True,
        delimiter=";",
        decimal=",",
    )
    rz_dwf_const = pd.read_csv(
        "swmm_output\latest_out_RZ_only_constant_DWF.csv",
        index_col=0,
        parse_dates=True,
        delimiter=";",
        decimal=",",
    )
    rz = pd.read_csv(
        "swmm_output\latest_out_RZ_out_WEST_precipitation.csv",
        index_col=0,
        parse_dates=True,
        delimiter=";",
        decimal=",",
    )
    rz_base = pd.read_csv(
        "swmm_output\latest_out_RZ_out_base.csv",
        index_col=0,
        parse_dates=True,
        delimiter=";",
        decimal=",",
    )

    ES = [es_dwf, es_base, es_dwf_const]
    RZ = [rz_dwf, rz_base, rz_dwf_const]

    labels = ["Scenario 1", "Scenario 2", "Scenario 3"]

    plt.figure(figsize=(10, 5))

    for i, ts in enumerate(ES):
        ts_fixed = ts.copy()
        ts_fixed.index = pd.to_datetime(ts_fixed.index)  # Ensure datetime index
        ts_fixed.index = ts_fixed.index.map(lambda dt: dt.replace(year=2024))
        ts_fixed = ts_fixed.loc["2024-07-01":"2024-08-01"]

        ts_fixed.H2O_sew.plot(label=labels[i])

    plt.title("Eindhoven city")
    plt.xlabel("Date")
    plt.ylabel("Catchment outflow [m3/d]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    for i, ts in enumerate(RZ):
        ts_fixed = ts.copy()
        ts_fixed.index = pd.to_datetime(ts_fixed.index)  # Ensure datetime index
        ts_fixed.index = ts_fixed.index.map(lambda dt: dt.replace(year=2024))
        ts_fixed = ts_fixed.loc["2024-07-01":"2024-08-01"]

        ts_fixed.H2O_sew.plot(label=labels[i])

    plt.title("Riool Zuid")
    plt.xlabel("Date")
    plt.ylabel("Catchment outflow [m3/d]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compare_wwtp_out():
    # all keys g/d
    constant = pd.read_csv(
        f"data\WEST\SWMM_input_constant_DWF\WWTP_output.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    start_date = pd.Timestamp("2024-01-01")
    constant["timestamp"] = start_date + pd.to_timedelta(
        constant.index.astype(float), unit="D"
    )
    constant.set_index("timestamp", inplace=True)

    dwf_only = pd.read_csv(
        f"data\WEST\SWMM_inputs_dwf_only\WWTP_output.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    start_date = pd.Timestamp("2024-01-01")
    dwf_only["timestamp"] = start_date + pd.to_timedelta(
        dwf_only.index.astype(float), unit="D"
    )
    dwf_only.set_index("timestamp", inplace=True)

    dwf_rain = pd.read_csv(
        f"data\WEST\SWMM_inputs_dwf_and_precipitation\WWTP_output.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    start_date = pd.Timestamp("2024-01-01")
    dwf_rain["timestamp"] = start_date + pd.to_timedelta(
        dwf_rain.index.astype(float), unit="D"
    )
    dwf_rain.set_index("timestamp", inplace=True)

    constant = constant.loc["2024-07-01":"2024-08-01"]
    dwf_only = dwf_only.loc["2024-07-01":"2024-08-01"]
    dwf_rain = dwf_rain.loc["2024-07-01":"2024-08-01"]

    fig = go.Figure()
    for key in constant.keys():
        fig.add_trace(
            go.Scatter(
                x=constant.index,
                y=constant[key].astype(float) * -1,
                mode="lines",
                name=f"{key}",
            )
        )
    fig.update_layout(title_text=f"Constant dwf")
    pio.show(fig, renderer="browser")

    fig = go.Figure()
    for key in dwf_only.keys():
        fig.add_trace(
            go.Scatter(
                x=dwf_only.index,
                y=dwf_only[key].astype(float) * -1,
                mode="lines",
                name=f"{key}",
            )
        )
    fig.update_layout(title_text=f"Only dwf")
    pio.show(fig, renderer="browser")

    fig = go.Figure()
    for key in dwf_rain.keys():
        fig.add_trace(
            go.Scatter(
                x=dwf_rain.index,
                y=dwf_rain[key].astype(float) * -1,
                mode="lines",
                name=f"{key}",
            )
        )
    fig.update_layout(title_text=f"Dwf and rain")
    pio.show(fig, renderer="browser")


def create_plots():
    # all keys g/d
    constant = pd.read_csv(
        f"data\WEST\SWMM_input_constant_DWF\WWTP_output.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    start_date = pd.Timestamp("2024-01-01")
    constant["timestamp"] = start_date + pd.to_timedelta(
        constant.index.astype(float), unit="D"
    )
    constant.set_index("timestamp", inplace=True)

    dwf_only = pd.read_csv(
        f"data\WEST\SWMM_inputs_dwf_only\WWTP_output.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    start_date = pd.Timestamp("2024-01-01")
    dwf_only["timestamp"] = start_date + pd.to_timedelta(
        dwf_only.index.astype(float), unit="D"
    )
    dwf_only.set_index("timestamp", inplace=True)

    dwf_rain = pd.read_csv(
        f"data\WEST\SWMM_inputs_dwf_and_precipitation\WWTP_output.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    start_date = pd.Timestamp("2024-01-01")
    dwf_rain["timestamp"] = start_date + pd.to_timedelta(
        dwf_rain.index.astype(float), unit="D"
    )
    dwf_rain.set_index("timestamp", inplace=True)

    constant_jul = constant.loc["2024-07-01":"2024-08-01"]
    dwf_only_jul = dwf_only.loc["2024-07-01":"2024-08-01"]
    dwf_rain_jul = dwf_rain.loc["2024-07-01":"2024-08-01"]

    constant_ext = constant.loc["2024-04-15":"2024-10-16"]
    dwf_only_ext = dwf_only.loc["2024-04-15":"2024-10-16"]
    dwf_rain_ext = dwf_rain.loc["2024-04-15":"2024-10-16"]

    WWTP_FULL = [dwf_only_ext, dwf_rain_ext, constant_ext]
    WWTP = [dwf_only_jul, dwf_rain_jul, constant_jul]
    labels = ["Scenario 1", "Scenaro 2", "Scenario 3"]
    values = [
        ".WWTP2river.Outflow(rH2O)",
        ".WWTP2river.Outflow(rNH4)",
        ".WWTP2river.Outflow(rO2)",
        ".WWTP2river.Outflow(rBOD2)",
        ".WWTP2river.Outflow(rBOD1)",
    ]

    for j, value in enumerate(values):
        if j == 0:
            plt.figure(figsize=(15, 7))
        else:
            plt.figure(figsize=(10, 7))
        for i, ts in enumerate(WWTP):
            ts[value].astype(float).abs().plot(label=labels[i])
        plt.xlabel("Date")
        plt.ylabel("Outflow [g/d]")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    for j, value in enumerate(values):
        i = 0
        for full, jul in zip(WWTP_FULL, WWTP):
            aaa = full.astype(float).resample("D").mean()
            bbb = jul.astype(float).resample("D").mean()
            print(f"{labels[i]}")
            print(f"{value}, ext: {aaa[value].astype(float).abs().sum() / 1000:.3f} kg")
            print(f"{value}, jul: {bbb[value].astype(float).abs().sum() / 1000:.3f} kg")
            i += 1


# Emptying in other file

# Storage in other file
