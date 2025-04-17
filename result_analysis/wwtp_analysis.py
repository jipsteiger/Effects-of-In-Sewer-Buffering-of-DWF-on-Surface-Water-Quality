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


def create_diff_table():
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

    base = pd.read_csv(
        f"data\WEST\SWMM_inputs_rtc_model\WWTP_output.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    start_date = pd.Timestamp("2024-01-01")
    base["timestamp"] = start_date + pd.to_timedelta(base.index.astype(float), unit="D")
    base.set_index("timestamp", inplace=True)

    rtc = pd.read_csv(
        f"data\WEST\SWMM_rtc_model\WWTP_output.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    start_date = pd.Timestamp("2024-01-01")
    rtc["timestamp"] = start_date + pd.to_timedelta(rtc.index.astype(float), unit="D")
    rtc.set_index("timestamp", inplace=True)

    base_jul = base.loc["2024-07-01":"2024-08-01"]
    rtc_jul = rtc.loc["2024-07-01":"2024-08-01"]

    base_ext = base.loc["2024-04-15":"2024-10-16"]
    rtc_ext = rtc.loc["2024-04-15":"2024-10-16"]

    # Use a proper variable name
    results = {
        "dwf_only_jul": [],
        "constant_jul": [],
        "dwf_rain_jul": [],
        "dwf_only_ext": [],
        "constant_ext": [],
        "dwf_rain_ext": [],
        "base_jul": [],
        "rtc_jul": [],
        "base_ext": [],
        "rtc_ext": [],
    }

    # Group DataFrames with labels
    dfs = {
        "dwf_only_jul": dwf_only_jul.astype(float),
        "constant_jul": constant_jul.astype(float),
        "dwf_rain_jul": dwf_rain_jul.astype(float),
        "dwf_only_ext": dwf_only_ext.astype(float),
        "constant_ext": constant_ext.astype(float).resample("D").mean(),
        "dwf_rain_ext": dwf_rain_ext.astype(float).resample("D").mean(),
        "base_jul": base_jul.astype(float).resample("D").mean(),
        "rtc_jul": rtc_jul.astype(float).resample("D").mean(),
        "base_ext": base_ext.astype(float).resample("D").mean(),
        "rtc_ext": rtc_ext.astype(float).resample("D").mean(),
    }

    keys = list(constant_jul.columns)
    index = []

    for key in keys:
        for name, df in dfs.items():
            if (".effluent.y" in key) and not ("y_Q" in key):
                value = (
                    df[key].astype(float) * df[".effluent.y_Q"].astype(float) * 24
                ).sum()
            elif (".effluent.y" in key) and ("y_Q" in key):
                value = (df[key].astype(float) * 24 * 1e6).sum()
            else:
                value = df[key].astype(float).sum()
            results[name].append(value)
        index.append(key)

    sums = pd.DataFrame(results, index=index).abs() / 1000
    sums["dif_jul_dwf"] = ((sums["constant_jul"] / sums["dwf_only_jul"]) - 1) * 100
    sums["dif_ext_dwf"] = ((sums["constant_ext"] / sums["dwf_only_ext"]) - 1) * 100
    sums["dif_jul_rtc"] = ((sums["rtc_jul"] / sums["base_jul"]) - 1) * 100
    sums["dif_ext_rtc"] = ((sums["rtc_ext"] / sums["base_ext"]) - 1) * 100
    diffs = sums[["dif_jul_dwf", "dif_ext_dwf", "dif_jul_rtc", "dif_ext_rtc"]]


def plot_maker():
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

    constant_jul = constant.loc["2024-07-01":"2024-08-01"].astype(float)
    dwf_only_jul = dwf_only.loc["2024-07-01":"2024-08-01"].astype(float)
    dwf_rain_jul = dwf_rain.loc["2024-07-01":"2024-08-01"].astype(float)

    constant_ext = constant.loc["2024-04-15":"2024-10-16"].astype(float)
    dwf_only_ext = dwf_only.loc["2024-04-15":"2024-10-16"].astype(float)
    dwf_rain_ext = dwf_rain.loc["2024-04-15":"2024-10-16"].astype(float)

    base = pd.read_csv(
        f"data\WEST\SWMM_inputs_rtc_model\WWTP_output.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    start_date = pd.Timestamp("2024-01-01")
    base["timestamp"] = start_date + pd.to_timedelta(base.index.astype(float), unit="D")
    base.set_index("timestamp", inplace=True)

    rtc = pd.read_csv(
        f"data\WEST\SWMM_rtc_model\WWTP_output.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    start_date = pd.Timestamp("2024-01-01")
    rtc["timestamp"] = start_date + pd.to_timedelta(rtc.index.astype(float), unit="D")
    rtc.set_index("timestamp", inplace=True)

    base_jul = base.loc["2024-07-01":"2024-08-01"].astype(float)
    rtc_jul = rtc.loc["2024-07-01":"2024-08-01"].astype(float)

    base_dry = base.loc["2024-07-27":"2024-07-30"].astype(float)
    rtc_dry = rtc.loc["2024-07-27":"2024-07-30"].astype(float)

    base_ext = base.loc["2024-04-15":"2024-10-16"].astype(float)
    rtc_ext = rtc.loc["2024-04-15":"2024-10-16"].astype(float)

    keys = [".effluent.y_NH", ".effluent.y_NO", ".effluent.y_BOD"]
    y_label = [
        "NH levels effluent [g/h]",
        "NO levels effluent [g/h]",
        "BOD levels effluent [g/h]",
    ]
    title = ["Effluent NH", "Effluent NO", "Effluent BOD"]
    dfs = {
        # "Normal DWF": dwf_only_jul.astype(float),
        # "Constant DWF": constant_jul.astype(float),
        # "dwf_rain_jul": dwf_rain_jul.astype(float),
        # "dwf_only_ext": dwf_only_ext.astype(float),
        # "constant_ext": constant_ext.astype(float),
        # "dwf_rain_ext": dwf_rain_ext.astype(float),
        # "Normal": base_jul.astype(float),
        # "RTC": rtc_jul.astype(float),
        "Normal": base_dry.astype(float),
        "RTC": rtc_dry.astype(float),
    }
    for i, key in enumerate(keys):
        plt.figure(figsize=(20, 4))
        for name, df in dfs.items():
            if (".effluent.y" in key) and not ("y_Q" in key):
                value = df[key].astype(float) * df[".effluent.y_Q"].astype(float)
                print(f"{name=}, {y_label[i]}: {value.sum()}")
                diff = (
                    df[key].astype(float).resample("D").mean()
                    * df[".effluent.y_Q"].astype(float).resample("D").mean()
                )
                print(f"{name=}, {y_label[i]}: {diff.sum() * 24 /1000}")

            elif (".effluent.y" in key) and ("y_Q" in key):
                value = df[key].astype(float)
            else:
                value = df[key].astype(float)
            plt.plot(df.index, value, label=f"{name}")
        plt.grid()
        plt.title(title[i])
        plt.xlabel("Date")
        plt.xticks(rotation=45)
        plt.ylabel(y_label[i])
        plt.legend()


def calc_perc(base, update):
    return ((update / base) - 1) * 100


def compare_wwtp_in():
    constant = pd.read_csv(
        f"data\WEST\SWMM_input_constant_DWF\wwtp_input.out.txt",
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
        f"data\WEST\SWMM_inputs_dwf_only\wwtp_input.out.txt",
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
        f"data\WEST\SWMM_inputs_dwf_and_precipitation\wwtp_input.out.txt",
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

    constant_jul = constant.loc["2024-07-01":"2024-08-01"].astype(float)
    dwf_only_jul = dwf_only.loc["2024-07-01":"2024-08-01"].astype(float)
    dwf_rain_jul = dwf_rain.loc["2024-07-01":"2024-08-01"].astype(float)

    constant_ext = constant.loc["2024-04-15":"2024-10-16"].astype(float)
    dwf_only_ext = dwf_only.loc["2024-04-15":"2024-10-16"].astype(float)
    dwf_rain_ext = dwf_rain.loc["2024-04-15":"2024-10-16"].astype(float)

    base = pd.read_csv(
        f"data\WEST\SWMM_inputs_rtc_model\wwtp_input.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    start_date = pd.Timestamp("2024-01-01")
    base["timestamp"] = start_date + pd.to_timedelta(base.index.astype(float), unit="D")
    base.set_index("timestamp", inplace=True)

    rtc = pd.read_csv(
        f"data\WEST\SWMM_rtc_model\wwtp_input.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    start_date = pd.Timestamp("2024-01-01")
    rtc["timestamp"] = start_date + pd.to_timedelta(rtc.index.astype(float), unit="D")
    rtc.set_index("timestamp", inplace=True)

    base_jul = base.loc["2024-07-01":"2024-08-01"].astype(float)
    rtc_jul = rtc.loc["2024-07-01":"2024-08-01"].astype(float)

    base_dry = base.loc["2024-07-27":"2024-07-30"].astype(float)
    rtc_dry = rtc.loc["2024-07-27":"2024-07-30"].astype(float)

    base_ext = base.loc["2024-04-15":"2024-10-16"].astype(float)
    rtc_ext = rtc.loc["2024-04-15":"2024-10-16"].astype(float)

    keys = [".WWTP_in.y_NH", ".WWTP_in.y_NO", ".WWTP_in.y_BOD", ".WWTP_in.y_Q"]
    y_label = [
        "NH levels inluent [g/h]",
        "NO levels inluent [g/h]",
        "BOD levels inluent [g/h]",
        "Q [m3/h]",
    ]
    title = ["Inluent NH", "Inluent NO", "Inluent BOD", "Inflow H2O"]
    dfs = {
        # "Normal DWF": dwf_only_jul.astype(float),
        # "Constant DWF": constant_jul.astype(float),
        # "dwf_rain_jul": dwf_rain_jul.astype(float),
        # "dwf_only_ext": dwf_only_ext.astype(float),
        # "constant_ext": constant_ext.astype(float),
        # "dwf_rain_ext": dwf_rain_ext.astype(float),
        # "Normal": base_jul.astype(float),
        # "RTC": rtc_jul.astype(float),
        "Normal": base_dry.astype(float),
        "RTC": rtc_dry.astype(float),
    }
    for i, key in enumerate(keys):
        plt.figure(figsize=(20, 4))
        for name, df in dfs.items():
            if (".WWTP_in.y" in key) and not ("y_Q" in key):
                value = df[key].astype(float) * df[".WWTP_in.y_Q"].astype(float)
                print(f"{name=}, {y_label[i]}: {value.sum()}")
                diff = (
                    df[key].astype(float).resample("D").mean()
                    * df[".WWTP_in.y_Q"].astype(float).resample("D").mean()
                )

            elif (".WWTP_in.y" in key) and ("y_Q" in key):
                value = df[key].astype(float)
            else:
                value = df[key].astype(float)
            plt.plot(df.index, value, label=f"{name}")

        plt.grid()
        plt.title(title[i])
        plt.xlabel("Date")
        plt.xticks(rotation=45)
        plt.ylabel(y_label[i])
        plt.legend()
