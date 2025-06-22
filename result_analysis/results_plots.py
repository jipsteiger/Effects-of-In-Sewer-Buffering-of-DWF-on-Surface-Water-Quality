import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import swmm_api as sa
from typing import List
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio
from emprical_sewer_wq import EmpericalSewerWQ
import plotly.io as pio
from data.concentration_curves import concentration_dict_ES, concentration_dict_RZ
from storage import Storage, RZ_storage
from storage import ConcentrationStorage
import os
from datetime import datetime
import datetime as dt
import matplotlib.dates as mdates
import numpy as np
import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename="plot_log.txt",
    filemode="w",
)


# Optional: seaborn colorblind-friendly palette
sns.set_palette("colorblind")


def nash_sutcliff(observed, modelled):
    obs_mean = observed.mean()
    NSE = 1 - ((observed - modelled) ** 2).sum() / ((observed - obs_mean) ** 2).sum()
    return NSE


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def plot(
    x_list: List[pd.Series],
    y_list: List[pd.Series],
    labels: List[str],
    x_label: str,
    y_label: str,
):
    for x, y, label in zip(x_list, y_list, labels):
        plt.plot(x.index, y.values, label=label)

    plt.xlabel(x_label, fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)

    # Format x-axis as mm-dd\nHH:MM
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List
import pandas as pd

pipes = [
    "Con_103",
    "Con_104",
    "Con_105",
    "Con_106",
    "Con_158",
    "Con_107",
    "Con_108",
    "Con_110",
    "Con_157",
    "Con_111",
    "Con_112",
    "Con_113",
    "Con_114",
    "Con_115",
    "Con_116",
    "Con_117",
    "Con_118",
    "Con_119",
    "Con_120",
    "Con_121",
    "Con_122",
    "Con_123",
    "Con_141",
    "Con_142",
    "Con_143",
    "Con_144",
    "Con_152",
    "Con_153",
    "Con_154",
    "Con_155",
    "Con_156",
    "Con_159",
    "Con_160",
    "Con_109",
]


def plot_two_regions_side_by_side(
    x_list1: List[pd.Series],
    y_list1: List[pd.Series],
    x_list2: List[pd.Series],
    y_list2: List[pd.Series],
    labels: List[str],
    x_label: str,
    y_label: str,
    region_titles: List[str],  # e.g., ["Region A", "Region B"]
):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharey=True)

    # First region
    for x, y, label in zip(x_list1, y_list1, labels):
        axes[0].plot(x.index, y.values, label=label)
    axes[0].set_title(region_titles[0])
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Second region
    for x, y, label in zip(x_list2, y_list2, labels):
        axes[1].plot(x.index, y.values, label=label)
    axes[1].set_title(region_titles[1])
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(y_label)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Format x-axis as mm-dd\nHH:MM
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        ax.tick_params(axis="x", rotation=0)

    # Single shared legend to the right of the second plot
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=9,
    )

    plt.tight_layout()  # Leave space on right for legend
    plt.show()


def model_setup_dwf():
    # Read models
    swmm = sa.read_out_file(r"data\SWMM\model_jip_WEST_regen.out").to_frame()
    west = pd.read_csv(
        r"data\WEST\WEST_modelRepository\Model_Dommel_Full\system_compare.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]

    # Setup WEST timestamps
    start_date = pd.Timestamp("2024-01-01")
    west["timestamp"] = start_date + pd.to_timedelta(west.index.astype(float), unit="D")
    west.set_index("timestamp", inplace=True)
    west.index = west.index.round("15min")

    # Extract flows
    ES_Q_swmm = swmm.link.P_eindhoven_out.flow.resample("15min").mean()
    ES_Q_swmm_in = swmm.node.pipe_ES.total_inflow.resample("15min").mean()
    RZ_Q_swmm = swmm.link.P_riool_zuid_out.flow.resample("15min").mean()
    ES_Q_west = west[".pipe_ES.Q_out"].astype(float) / 24 / 3600
    RZ_Q_west = west[".pipe_RZ.Q_out"].astype(float) / 24 / 3600

    # Align ES flows
    ES_Q_swmm_aligned, ES_Q_west_aligned = ES_Q_swmm.align(ES_Q_west, join="inner")
    EQ_Q_swmm_in_aligned, _ = ES_Q_swmm_in.align(ES_Q_west, join="inner")
    RZ_Q_swmm_aligned, RZ_Q_west_aligned = RZ_Q_swmm.align(RZ_Q_west, join="inner")

    # Crop to analysis period
    start, end = "2024-06-27", "2024-06-29"
    ES_Q_swmm = ES_Q_swmm_aligned.loc[start:end]
    ES_Q_west = ES_Q_west_aligned.loc[start:end]
    RZ_Q_swmm = RZ_Q_swmm_aligned.loc[start:end]
    RZ_Q_west = RZ_Q_west_aligned.loc[start:end]
    ES_Q_swmm_in = ES_Q_swmm_in.loc[start:end]

    # Plot ES
    nash_sutcliff(ES_Q_west.values, ES_Q_swmm.values)
    rmse(ES_Q_west.values, ES_Q_swmm.values)
    plot(
        x_list=[ES_Q_swmm, ES_Q_west, ES_Q_swmm_in],
        y_list=[ES_Q_swmm, ES_Q_west, ES_Q_swmm_in],
        labels=["SWMM", "WEST", "SWMM In"],
        x_label="Date and time",
        y_label="Discharge $Q$ [m³/s]",
    )

    # Plot RZ
    nash_sutcliff(RZ_Q_west.values, RZ_Q_swmm.values)
    rmse(RZ_Q_west.values, RZ_Q_swmm.values)
    plot(
        x_list=[RZ_Q_swmm, RZ_Q_west],
        y_list=[RZ_Q_swmm, RZ_Q_west],
        labels=["SWMM", "WEST"],
        x_label="Date and time",
        y_label="Discharge $Q$ [m³/s]",
    )

    # Shift SWMM by -2.5h and align
    RZ_Q_swmm_shifted = RZ_Q_swmm.shift(freq="-120min")
    RZ_Q_swmm_aligned, RZ_Q_west_aligned = RZ_Q_swmm_shifted.align(
        RZ_Q_west, join="inner"
    )

    # Plot shifted
    nash_sutcliff(RZ_Q_west_aligned.values, RZ_Q_swmm_aligned.values)
    rmse(RZ_Q_west_aligned.values, RZ_Q_swmm_aligned.values)
    plot(
        x_list=[RZ_Q_swmm_aligned, RZ_Q_west_aligned],
        y_list=[RZ_Q_swmm_aligned, RZ_Q_west_aligned],
        labels=["SWMM (shifted)", "WEST"],
        x_label="Date and time",
        y_label="Discharge $Q$ [m³/s]",
    )


def model_setup_wwf():
    swmm = sa.read_out_file(r"data\SWMM\model_jip_WEST_regen.out").to_frame()
    west = pd.read_csv(
        r"data\WEST\WEST_modelRepository\Model_Dommel_Full\system_compare.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]

    # Setup WEST timestamps
    start_date = pd.Timestamp("2024-01-01")
    west["timestamp"] = start_date + pd.to_timedelta(west.index.astype(float), unit="D")
    west.set_index("timestamp", inplace=True)
    west.index = west.index.round("15min")

    # Extract flows
    ES_Q_swmm = swmm.link.P_eindhoven_out.flow.resample("15min").mean()
    RZ_Q_swmm = swmm.link.P_riool_zuid_out.flow.resample("15min").mean()
    ES_Q_west = west[".pipe_ES.Q_out"].astype(float) / 24 / 3600
    RZ_Q_west = west[".pipe_RZ.Q_out"].astype(float) / 24 / 3600

    # Align ES flows
    ES_Q_swmm_aligned, ES_Q_west_aligned = ES_Q_swmm.align(ES_Q_west, join="inner")
    RZ_Q_swmm_aligned, RZ_Q_west_aligned = RZ_Q_swmm.align(RZ_Q_west, join="inner")

    # Crop to analysis period
    start, end = "2024-06-01", "2024-06-10"
    ES_Q_swmm = ES_Q_swmm_aligned.loc[start:end]
    ES_Q_west = ES_Q_west_aligned.loc[start:end]
    RZ_Q_swmm = RZ_Q_swmm_aligned.loc[start:end]
    RZ_Q_west = RZ_Q_west_aligned.loc[start:end]

    # Plot ES
    nash_sutcliff(ES_Q_west.values, ES_Q_swmm.values)
    plot(
        x_list=[ES_Q_swmm, ES_Q_west],
        y_list=[ES_Q_swmm, ES_Q_west],
        labels=["SWMM", "WEST"],
        x_label="Date and time",
        y_label="Discharge $Q$ [m³/s]",
    )

    # Plot RZ
    nash_sutcliff(RZ_Q_west.values, RZ_Q_swmm.values)
    plot(
        x_list=[RZ_Q_swmm, RZ_Q_west],
        y_list=[RZ_Q_swmm, RZ_Q_west],
        labels=["SWMM", "WEST"],
        x_label="Date and time",
        y_label="Discharge $Q$ [m³/s]",
    )

    # Shift SWMM by -2.5h and align
    RZ_Q_swmm_shifted = RZ_Q_swmm.shift(freq="-120min")
    RZ_Q_swmm_aligned, RZ_Q_west_aligned = RZ_Q_swmm_shifted.align(
        RZ_Q_west, join="inner"
    )

    # Plot shifted
    nash_sutcliff(RZ_Q_west_aligned.values, RZ_Q_swmm_aligned.values)
    plot(
        x_list=[RZ_Q_swmm_aligned, RZ_Q_west_aligned],
        y_list=[RZ_Q_swmm_aligned, RZ_Q_west_aligned],
        labels=["SWMM (shifted)", "WEST"],
        x_label="Date and time",
        y_label="Discharge $Q$ [m³/s]",
    )


def model_creation_csos():
    df_west = pd.read_csv(
        r"data\WEST\WEST_modelRepository\Model_Dommel_Full\CSO_AND_INFLOW.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    start_date = pd.Timestamp("2024-01-01")
    df_west["timestamp"] = start_date + pd.to_timedelta(
        df_west.index.astype(float), unit="D"
    )
    df_west.set_index("timestamp", inplace=True)

    output = sa.SwmmOutput(rf"data\SWMM\model_jip_WEST_regen.out").to_frame()

    swmm_csos = {
        "ES": ["cso_ES_1"],
        "GB": ["cso_gb_136"],
        "GE": ["cso_Geldrop", "cso_gb127"],
        "TL": ["cso_RZ"],
        "RZ": [
            "cso_AALST",
            "cso_c_123",
            "cso_c_122",
            "cso_c_119_1",
            "cso_c_119_2",
            "cso_c_119_3",
            "cso_c_112",
            "cso_c_99",
        ],
    }

    Q_keys = df_west.keys()
    cso_ES_1_keys = [key for key in Q_keys if "Ein" in key]

    cso_Geldrop_keys = [
        key
        for key in Q_keys
        if any(substring in key for substring in ["Mierlo", "Geldrop"])
    ]

    cso_gb_136_keys = [
        key
        for key in Q_keys
        if any(substring in key for substring in ["Heeze2", "Leende"])
    ]

    cso_RZ_keys = [key for key in Q_keys if "Collse_Molen" in key]

    cso_AALST_keys = [
        key
        for key in Q_keys
        if any(
            substring in key
            for substring in [
                "Aalst",
                "Valkenswaard",
                "Dom",
                "Westerhoven",
                "Bergeijk",
            ]
        )
        and not ".Aalst_gemaal" in key
    ] + [".Waalre.Q_in", ".Aalst.Q_in"]

    west_csos = {
        "ES": cso_ES_1_keys,
        "GB": cso_gb_136_keys,
        "GE": cso_Geldrop_keys,
        "TL": cso_RZ_keys,
        "RZ": cso_AALST_keys,
    }
    start, end = "2024-04-15", "2024-10-15"
    df_west = df_west.loc[start:end]
    fig = go.Figure()
    for key in swmm_csos.keys():
        swmm_values = 0
        for swmm_cso in swmm_csos[key]:
            swmm_values += output.node[swmm_cso].total_inflow.values * 3600

        west_values = df_west[west_csos[key][:]].astype(float).sum(axis=1)

        print(f"swmm {key} sum={sum(swmm_values)}, west {key} sum={sum(west_values)}")
        fig.add_trace(
            go.Scatter(
                x=output.index,
                y=swmm_values,
                mode="lines",
                name=f"SWMM cso flow {key}",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=west_values.index,
                y=west_values.values,
                mode="lines",
                name=f"WEST cso flow {key}",
            )
        )

    fig.update_layout(
        title_text="Total CSO flow per catchment comparison",
        xaxis_title="Date",
        yaxis_title="Flow [m3/h]",
        legend_title="",
        bargap=0.1,  # Reducing gap between bars for better visibility
    )
    pio.show(fig, renderer="browser")
    # Keep old table I cant be bothered with rerunning west
    # Catchment     SWMM    WEST
    # ES            9       9
    # Geldrop       11      11
    # Mierlo        4       3
    # Aalst         16      22
    # Would need to be updated tho if swmm change (but then onyl sswmm values are adjusted i quess)


def wq_model():
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

    model_result_ES = pd.read_csv(
        "output_effluent\compare_ES_WQ_model_check.Effluent.csv", index_col=0
    )
    model_result_ES.index = df_west.index

    model_result_RZ = pd.read_csv(
        "output_effluent\compare_rz_WQ_model_check.Effluent.csv", index_col=0
    )
    model_result_RZ.index = df_west.index

    df_west = df_west.loc["2024-07-10":"2024-07-24"]
    model_result_ES = model_result_ES.loc["2024-07-10":"2024-07-24"]
    model_result_RZ = model_result_RZ.loc["2024-07-10":"2024-07-24"]

    xlist = []
    ylist = []
    labels = []
    labels = [
        "Original COD particulate",
        "Original COD soluable",
        "Original NH4",
        "Original PO4",
        "Original # TSS",
        "Recreated COD particulate",
        "Recreated COD soluable",
        "Recreated NH4",
        "Recreated PO4",
        "Recreated # TSS",
    ]
    for key in df_west.keys():
        if (
            "_out.Outflow" in key
            and not ".NS" in key
            and "ES" in key
            and not "H2O" in key
            and ('NH4' in key or 'PO4' in key)
        ):
            xlist.append(df_west)
            ylist.append(abs(df_west[key].astype(float)))
            # labels.append(f"Original model {key}")

    for key in model_result_ES.keys():
        if (not "H2O" in key and ('NH4' in key or 'PO4' in key)):
            xlist.append(model_result_ES)
            ylist.append(abs(model_result_ES[key].astype(float)))
            # labels.append(f"Script model {key}")

    NSE_COD_p = nash_sutcliff(
        abs(df_west[".ES_out.Outflow(COD_part)"].astype(float)),
        abs(model_result_ES["COD_part"].astype(float)),
    )
    NSE_COD_s = nash_sutcliff(
        abs(df_west[".ES_out.Outflow(COD_sol)"].astype(float)),
        abs(model_result_ES["COD_sol"].astype(float)),
    )
    NSE_NH4 = nash_sutcliff(
        abs(df_west[".ES_out.Outflow(NH4_sew)"].astype(float)),
        abs(model_result_ES["NH4_sew"].astype(float)),
    )
    NSE_PO4 = nash_sutcliff(
        abs(df_west[".ES_out.Outflow(PO4_sew)"].astype(float)),
        abs(model_result_ES["PO4_sew"].astype(float)),
    )
    NSE_TSS = nash_sutcliff(
        abs(df_west[".ES_out.Outflow(X_TSS_sew)"].astype(float)),
        abs(model_result_ES["X_TSS_sew"].astype(float)),
    )

    RMSE_COD_p = rmse(
        abs(df_west[".ES_out.Outflow(COD_part)"].astype(float)),
        abs(model_result_ES["COD_part"].astype(float)),
    )
    RMSE_COD_s = rmse(
        abs(df_west[".ES_out.Outflow(COD_sol)"].astype(float)),
        abs(model_result_ES["COD_sol"].astype(float)),
    )
    RMSE_NH4 = rmse(
        abs(df_west[".ES_out.Outflow(NH4_sew)"].astype(float)),
        abs(model_result_ES["NH4_sew"].astype(float)),
    )
    RMSE_PO4 = rmse(
        abs(df_west[".ES_out.Outflow(PO4_sew)"].astype(float)),
        abs(model_result_ES["PO4_sew"].astype(float)),
    )
    RMSE_TSS = rmse(
        abs(df_west[".ES_out.Outflow(X_TSS_sew)"].astype(float)),
        abs(model_result_ES["X_TSS_sew"].astype(float)),
    )

    # plt.figure(figsize=(15, 8))
    # plot(xlist, ylist, labels, "Date and time", "Catchment effluent load [g/d]")
    xlist1 = xlist
    ylist1 = ylist

    xlist = []
    ylist = []
    labels = []
    labels = [
        # "Original COD particulate",
        # "Original COD soluable",
        "Original NH4",
        "Original PO4",
        # "Original # TSS",
        # "Recreated COD particulate",
        # "Recreated COD soluable",
        "Recreated NH4",
        "Recreated PO4",
        # "Recreated # TSS",
    ]
    for key in df_west.keys():
        if (
            "_out.Outflow" in key
            and not ".NS" in key
            and "RZ" in key
            and not "H2O" in key
            and ('NH4' in key or 'PO4' in key)
        ):
            xlist.append(df_west)
            ylist.append(abs(df_west[key].astype(float)))
            # labels.append(f"Original model {key}")

    for key in model_result_RZ.keys():
        if (not "H2O" in key and ('NH4' in key or 'PO4' in key)):
            xlist.append(model_result_RZ)
            ylist.append(abs(model_result_RZ[key].astype(float)))
            # labels.append(f"Script model {key}")

    NSE_COD_p = nash_sutcliff(
        abs(df_west[".RZ_out.Outflow(COD_part)"].astype(float)),
        abs(model_result_RZ["COD_part"].astype(float)),
    )
    NSE_COD_s = nash_sutcliff(
        abs(df_west[".RZ_out.Outflow(COD_sol)"].astype(float)),
        abs(model_result_RZ["COD_sol"].astype(float)),
    )
    NSE_NH4 = nash_sutcliff(
        abs(df_west[".RZ_out.Outflow(NH4_sew)"].astype(float)),
        abs(model_result_RZ["NH4_sew"].astype(float)),
    )
    NSE_PO4 = nash_sutcliff(
        abs(df_west[".RZ_out.Outflow(PO4_sew)"].astype(float)),
        abs(model_result_RZ["PO4_sew"].astype(float)),
    )
    NSE_TSS = nash_sutcliff(
        abs(df_west[".RZ_out.Outflow(X_TSS_sew)"].astype(float)),
        abs(model_result_RZ["X_TSS_sew"].astype(float)),
    )

    plt.figure()
    plot(xlist1, ylist, labels, "Date and time", "Catchment effluent load [g/d]")
    plot_two_regions_side_by_side(
        xlist1,
        ylist1,
        xlist,
        ylist,
        labels,
        "Date and time",
        "Effluent load [g/d]",
        region_titles=["Eindhoven City catchment", "Riool Zuid catchment"],
    )


def wq_model_flow_param():
    class Q_95_norm_ES:
        H_0 = 1.04
        H_1 = 1.00
        H_2 = 0.94
        H_3 = 0.87
        H_4 = 0.80
        H_5 = 0.75
        H_6 = 0.72
        H_7 = 0.74
        H_8 = 0.84
        H_9 = 0.97
        H_10 = 1.06
        H_11 = 1.10
        H_12 = 1.11
        H_13 = 1.14
        H_14 = 1.15
        H_15 = 1.14
        H_16 = 1.10
        H_17 = 1.09
        H_18 = 1.06
        H_19 = 1.07
        H_20 = 1.08
        H_21 = 1.09
        H_22 = 1.08
        H_23 = 1.06

    # Extract values
    hours = list(range(24))
    average_flow_ES = 0.819
    values_ES = [getattr(Q_95_norm_ES, f"H_{h}") * average_flow_ES for h in hours]

    swmm = sa.read_out_file(
        r"data\SWMM\model_jip_no_RTC_no_rain_constant.out"
    ).to_frame()
    inflow_ES = swmm.node.pipe_ES.total_inflow["2023-04-25"].resample("h").mean().values
    inflow_norm_ES = inflow_ES / inflow_ES.mean()
    transformed_ES = np.round(inflow_norm_ES, 2) + 0.05

    # Plot
    plt.figure(figsize=(10, 5))
    plt.scatter(hours, values_ES, color="blue", label="WEST Q95 Hourly Average Oatterb")
    plt.plot(hours, inflow_ES, color="orange", label="SWMM Tank Inflow")
    plt.scatter(
        hours,
        transformed_ES * inflow_ES.mean(),
        color="green",
        label="Adjusted Hourly Pattern",
    )
    plt.axhline(
        average_flow_ES,
        color="k",
        linestyle="--",
        label=f"Daily Average Q95 Flow = {average_flow_ES:.2f}",
    )
    plt.axhline(
        inflow_ES.mean(),
        color="k",
        linestyle="-.",
        label=f"Average SWMM Tank Inflow = {inflow_ES.mean():.2f}",
    )
    plt.xticks(hours)
    plt.xlabel("Hour of Day")
    plt.ylabel("Flow [m3/s]")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    class Q_95_norm_RZ:
        H_0 = 1.04
        H_1 = 1.00
        H_2 = 0.94
        H_3 = 0.87
        H_4 = 0.80
        H_5 = 0.75
        H_6 = 0.72
        H_7 = 0.74
        H_8 = 0.84
        H_9 = 0.97
        H_10 = 1.06
        H_11 = 1.10
        H_12 = 1.11
        H_13 = 1.14
        H_14 = 1.15
        H_15 = 1.14
        H_16 = 1.10
        H_17 = 1.09
        H_18 = 1.06
        H_19 = 1.07
        H_20 = 1.08
        H_21 = 1.09
        H_22 = 1.08
        H_23 = 1.06

    # Extract values
    hours = list(range(24))
    average_flow_RZ = 0.6805
    values_RZ = [getattr(Q_95_norm_RZ, f"H_{h}") * average_flow_RZ for h in hours]

    swmm = sa.read_out_file(
        r"data\SWMM\model_jip_no_RTC_no_rain_constant.out"
    ).to_frame()
    RZ_in = swmm.node["Nod_112"].total_inflow + swmm.node["Nod_104"].total_inflow
    inflow_RZ = RZ_in["2023-04-25"].resample("h").mean().values
    inflow_norm_RZ = inflow_RZ / inflow_RZ.mean()
    transformed_RZ = np.round(inflow_norm_RZ, 2) + 0.05

    # Plot
    plt.figure(figsize=(10, 5))
    plt.scatter(hours, values_RZ, color="blue", label="WEST Q95 Hourly Average Pattern")
    plt.plot(hours, inflow_RZ, color="orange", label="SWMM Tank Inflow")
    plt.scatter(
        hours,
        transformed_RZ * inflow_RZ.mean(),
        color="green",
        label="Adjusted Hourly Pattern",
    )
    plt.axhline(
        average_flow_RZ,
        color="k",
        linestyle="--",
        label=f"Daily Average Q95 Flow = {average_flow_RZ:.2f}",
    )
    plt.axhline(
        inflow_RZ.mean(),
        color="k",
        linestyle="-.",
        label=f"Average SWMM Tank Inflow = {inflow_RZ.mean():.2f}",
    )
    plt.xticks(hours)
    plt.xlabel("Hour of Day")
    plt.ylabel("Flow [m3/s]")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def WQ_model_location():
    # DOESNT WORK
    swmm = sa.read_out_file(r"data\SWMM\model_jip_WEST_regen.out").to_frame()

    start, end = "2024-06-27", "2024-06-30"
    swmm = swmm.loc[start:end]

    ES_out = swmm.link.P_eindhoven_out.flow * 3600 * 24
    RZ_out = swmm.link.P_riool_zuid_out.flow * 3600 * 24
    ES_storage_FD = Storage(165000)
    RZ_storage_FD = RZ_storage(
        35721,
        pipes=[
            "Con_103",
            "Con_104",
            "Con_105",
            "Con_106",
            "Con_158",
            "Con_107",
            "Con_108",
            "Con_110",
            "Con_157",
            "Con_111",
            "Con_112",
            "Con_113",
            "Con_114",
            "Con_115",
            "Con_116",
            "Con_117",
            "Con_118",
            "Con_119",
            "Con_120",
            "Con_121",
            "Con_122",
            "Con_123",
            "Con_141",
            "Con_142",
            "Con_143",
            "Con_144",
            "Con_152",
            "Con_153",
            "Con_154",
            "Con_155",
            "Con_156",
            "Con_159",
            "Con_160",
            "Con_109",
        ],
    )

    ES_in = swmm.node["pipe_ES"].total_inflow * 3600 * 24
    RZ_in = (
        (swmm.node["Nod_112"].total_inflow + swmm.node["Nod_104"].total_inflow)
        * 3600
        * 24
    )

    ESConcentrationStorage_after = ConcentrationStorage()
    RZConcentrationStorage_after = ConcentrationStorage()

    pipes = [
        "Con_103",
        "Con_104",
        "Con_105",
        "Con_106",
        "Con_158",
        "Con_107",
        "Con_108",
        "Con_110",
        "Con_157",
        "Con_111",
        "Con_112",
        "Con_113",
        "Con_114",
        "Con_115",
        "Con_116",
        "Con_117",
        "Con_118",
        "Con_119",
        "Con_120",
        "Con_121",
        "Con_122",
        "Con_123",
        "Con_141",
        "Con_142",
        "Con_143",
        "Con_144",
        "Con_152",
        "Con_153",
        "Con_154",
        "Con_155",
        "Con_156",
        "Con_159",
        "Con_160",
        "Con_109",
    ]
    WQ_ES_after = EmpericalSewerWQ(
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
    WQ_RZ_after = EmpericalSewerWQ(
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
    WQ_ES_before = EmpericalSewerWQ(
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
    WQ_RZ_before = EmpericalSewerWQ(
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

    ES_conc_after_fix = []
    RZ_conc_after_fix = []
    ES_conc_before_fix = []
    RZ_conc_before_fix = []
    timestamps = []
    for index, Q_ES_out, Q_rz_out, Q_ES_in, Q_RZ_in in zip(
        swmm.index, ES_out, RZ_out, ES_in, RZ_in
    ):
        ES_vol = swmm.node.pipe_ES.volume[index]
        # swmm.link.volume.loc[index, pipes].sum()
        RZ_vol = sum([swmm.link[pipe].volume[index] for pipe in pipes])

        ES_storage_FD.update_stored_volume(ES_vol)
        RZ_storage_FD.update_stored_volume(RZ_vol)

        WQ_ES_after.update(index, Q_ES_in, ES_storage_FD.FD())
        WQ_RZ_after.update(index, Q_RZ_in, RZ_storage_FD.FD())

        WQ_ES_before.update(index, Q_ES_out, ES_storage_FD.FD())
        WQ_RZ_before.update(index, Q_rz_out, RZ_storage_FD.FD())

        RZ_pollutant_after = WQ_RZ_after.get_latest_log()
        ES_pollutant_after = WQ_ES_after.get_latest_log()
        RZ_pollutant_before = WQ_RZ_before.get_latest_log()
        ES_pollutant_before = WQ_ES_before.get_latest_log()

        ESConcentrationStorage_after.update_in(
            Q_ES_in / 24 / 3600, ES_pollutant_after, ES_vol
        )
        RZConcentrationStorage_after.update_in(
            Q_RZ_in / 24 / 3600, RZ_pollutant_after, RZ_vol
        )

        ES_conc_out_after = ESConcentrationStorage_after.update_out(
            Q_ES_out / 24 / 3600, ES_storage_FD.FD(), ES_vol
        )
        RZ_conc_out_after = RZConcentrationStorage_after.update_out(
            Q_rz_out / 24 / 3600, RZ_storage_FD.FD(), RZ_vol
        )
        print(index)
        timestamps.append(index)
        ES_conc_after_fix.append(ES_conc_out_after)
        RZ_conc_after_fix.append(RZ_conc_out_after)
        ES_conc_before_fix.append(ES_pollutant_before)
        RZ_conc_before_fix.append(RZ_pollutant_before)

    # Convert to DataFrames
    df_ES_fixed = pd.DataFrame(ES_conc_after_fix, index=timestamps)
    df_RZ_fixed = pd.DataFrame(RZ_conc_after_fix, index=timestamps)

    df_ES_old = pd.DataFrame(ES_conc_before_fix, index=timestamps)
    df_RZ_old = pd.DataFrame(RZ_conc_before_fix, index=timestamps)

    for key in df_ES_fixed.keys():
        if not (("FD" in key) or ("H2O" in key)):
            df_ES_fixed[f"{key}_conc"] = abs(df_ES_fixed[key] / df_ES_fixed["H2O_sew"])
            df_RZ_fixed[f"{key}_conc"] = abs(df_RZ_fixed[key] / df_RZ_fixed["H2O_sew"])
            df_ES_old[f"{key}_conc"] = abs(df_ES_old[key] / df_ES_old["H2O_sew"])
            df_RZ_old[f"{key}_conc"] = abs(df_RZ_old[key] / df_RZ_old["H2O_sew"])

    # Save to CSV files
    df_ES_fixed.to_csv("ES_pollutant_fixed.csv")
    df_RZ_fixed.to_csv("RZ_pollutant_fixed.csv")
    df_ES_old.to_csv("ES_pollutant_not_fixed.csv")
    df_RZ_old.to_csv("RZ_pollutant_not_fixed.csv")

    key = "NH4_sew"
    plt.figure()
    df_ES_old[key].plot(label="Conc old")
    df_ES_fixed[key].plot(label="Conc fixed")
    plt.legend()

    plt.figure()
    df_ES_old[f"{key}_conc"].plot(label="Conc old")
    df_ES_fixed[f"{key}_conc"].plot(label="Conc fixed")


def WQ_model_location2():
    from simulation import Simulation
    from postprocess import PostProcess

    SUFFIX = "No_RTC_no_rain_WQ_wrong_loc"
    MODEL_NAME = "model_jip_no_rtc_no_rain_WQ_wrong_loc"
    simulation = Simulation(
        model_path=rf"data\SWMM\{MODEL_NAME}.inp",
        step_size=300,
        report_start=dt.datetime(year=2023, month=4, day=15),
        start_time=dt.datetime(year=2023, month=4, day=15),
        end_time=dt.datetime(year=2023, month=5, day=30),
        virtual_pump_max=10,
    )
    simulation.start_simulation()
    postprocess = PostProcess(model_name=MODEL_NAME)
    postprocess.create_outfall_txt_concentrate(suffix=SUFFIX, specific_version="no_RTC")

    labels = {
        "H2O_sew": "Discharge [g/d]",
        "NH4_sew": "NH4 Load [g/d]",
        "PO4_sew": "PO4 Load [g/d]",
        "COD_sol": "Solluable COD Load [g/d]",
        "X_TSS_sew": "Total Suspended Solids Load [g/d]",
        "COD_part": "Particulate COD Load [g/d]",
    }
    labels_conc = {
        "H2O_sew": "Discharge [g/d]",
        "NH4_sew": "NH4 Concentration [g/g]",
        "PO4_sew": "PO4 Concentration [g/g]",
        "COD_sol": "Solluable COD Concentration [g/g]",
        "X_TSS_sew": "Total Suspended Solids Concentration [g/g]",
        "COD_part": "Particulate COD Concentration [g/g]",
    }

    start_date = pd.Timestamp("2024-01-01")
    beforestorage = pd.read_csv(
        rf"output_swmm\06-01_15-48_out_RZ_No_RTC_no_rain_WQ_wrong_loc.csv",
        decimal=",",
        delimiter=";",
        index_col=0,
    )
    beforestorage["timestamp"] = start_date + pd.to_timedelta(
        beforestorage.index.astype(float), unit="D"
    )
    beforestorage.set_index("timestamp", inplace=True)
    beforestorage.index = beforestorage.index.round("15min")

    afterstorage = pd.read_csv(
        rf"output_swmm\06-01_16-46_out_RZ_No_RTC_no_rain.csv",
        decimal=",",
        delimiter=";",
        index_col=0,
    )
    afterstorage["timestamp"] = start_date + pd.to_timedelta(
        afterstorage.index.astype(float), unit="D"
    )
    afterstorage.set_index("timestamp", inplace=True)
    afterstorage.index = afterstorage.index.round("15min")

    # Define your time window
    start_date = pd.Timestamp("2024-04-15")
    end_date = pd.Timestamp("2024-04-20")

    scenarios = {
        "Water quality model placed after the storage unit": afterstorage,
        "Water quality model placed before the storage unit": beforestorage,
    }

    for key in list(afterstorage.keys()):
        if not ("FD" in key or "Q_out" in key):
            # Filter by time window and store filtered versions
            filtered_scenarios = {
                key: df.loc[start_date:end_date] for key, df in scenarios.items()
            }

            # Determine subplot layout based on whether to include concentration
            include_conc = "H2O" not in key

            # Create figure and axes
            if include_conc:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
            else:
                fig, ax1 = plt.subplots(figsize=(14, 6))

            # --- Absolute values plot ---
            for label, df in filtered_scenarios.items():
                if key in df.columns:
                    ax1.plot(df.index, abs(df[key]), label=label)
                else:
                    print(f"Column '{key}' not found in scenario '{label}'")

            ax1.set_ylabel(labels[key])
            ax1.legend(loc=1)
            ax1.grid(True)

            # --- Concentration plot (only if applicable) ---
            if include_conc:
                for label, df in filtered_scenarios.items():
                    if key in df.columns and "H2O_sew" in df.columns:
                        ax2.plot(df.index, df[key] / df["H2O_sew"], label=label)
                    else:
                        print(
                            f"Column '{key}' or 'H2O_sew' not found in scenario '{label}'"
                        )

                ax2.set_ylabel(labels_conc[key])
                ax2.legend(loc=1)
                ax2.grid(True)
                ax1.set_xlabel("Date")
                ax2.set_xlabel("Date")
            else:
                ax1.set_xlabel("Date")

            # Final layout
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            for label in ax1.get_xticklabels():
                label.set_rotation(45)
            ax1.legend(loc=1, frameon=True, facecolor="white", framealpha=0.8)
            if include_conc:
                ax2.legend(loc=4, frameon=True, facecolor="white", framealpha=0.8)
                for label in ax2.get_xticklabels():
                    label.set_rotation(45)
            # plt.savefig(f"{key}")
            plt.show()


def precipitation_data_forecast():
    path_ensemble = rf"data\precipitation\raw_data\zip_raw_harmonie"

    files = [
        f
        for f in os.listdir(path_ensemble)
        if (
            os.path.isfile(os.path.join(path_ensemble, f))
            and f.endswith(".nc")
            # and any(f"_0{horizon}.nc" in f for horizon in forecast_horizon)
        )
    ]

    from collections import defaultdict, Counter
    import re

    # Dictionary to store counts
    counts = defaultdict(int)

    # Extract date part and count
    for file in files:
        match = re.search(r"_(\d{10})_", file)
        if match:
            datetime_str = match.group(1)
            date_str = datetime_str[:8]  # Extract yyyymmdd
            counts[date_str] += 1
    # Plot all counts (or just filtered_counts for <188)
    dates = sorted(counts.keys())
    counts_per_day = [counts[date] for date in dates]

    # Convert to datetime
    dates_dt = [datetime.strptime(date, "%Y%m%d") for date in dates]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates_dt, counts_per_day, marker="o", linestyle="-", color="steelblue")

    # Format x-axis

    plt.xlabel("Date")
    plt.ylabel("File Count")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    ######## Plot of ensemble vs actual
    forecasts = pd.read_csv(
        r"data\precipitation\csv_forecasts\forecast_data.csv", index_col=0
    )
    forecasts["date"] = pd.to_datetime(forecasts["date"])
    forecasts["date_of_forecast"] = pd.to_datetime(forecasts["date_of_forecast"])
    forecasts["ensembles"] = forecasts["ensembles"].apply(
        lambda s: [float(x) for x in s.strip("[]").split()]
    )

    start_time = "2024-05-13 12:00:00"
    end_time = "2024-05-14 00:00:00"
    current_time = pd.to_datetime(start_time)  # 6-hour intervals
    upperbound = 48  # hours
    upperbound_time = current_time + dt.timedelta(hours=upperbound)

    filtered_forecast = forecasts[
        (forecasts.date == current_time)
        & (forecasts.date_of_forecast <= upperbound_time)
    ]

    grouped = filtered_forecast.groupby(["region", "date_of_forecast"])[
        "ensembles"
    ].apply(list)

    precipitation = pd.read_csv(
        r"data\precipitation\csv_selected_area_euradclim\2024_5_min_precipitation_data.csv",
        index_col=0,
        parse_dates=True,
    )
    precipitation = precipitation.resample("h").sum()
    precipitation = precipitation.loc[start_time:end_time, "RZ2"]

    ES = grouped["RZ2"].reset_index()
    ES["timestamp"] = pd.to_datetime(ES["date_of_forecast"])
    ES = ES.sort_values("timestamp")

    ensemble_data = ES["ensembles"]
    positions = mdates.date2num(ES["timestamp"])

    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(
        precipitation.index,
        precipitation.values,
        color="lightblue",
        lw=4,
        label="Precipitation",
    )
    ax.boxplot(
        ensemble_data,
        positions=positions,
        widths=0.025,
        patch_artist=True,  # Allows filling boxes
        showfliers=False,  # Optionally hide outliers if they clutter
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(facecolor="lightblue", alpha=0.7, edgecolor="gray"),
        whiskerprops=dict(color="gray", linestyle="--"),
        capprops=dict(color="gray"),
    )

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.xticks(rotation=90)
    ax.set_xlabel("Date and time")
    ax.set_ylabel("Precipitation [mm]")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    ax.set_xlim(min(positions) - 0.1, max(positions) + 0.1)
    fig.tight_layout()
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.show()


def system_character():
    catchment = "ES"
    labels = {
        "H2O_sew": "Discharge [g/d]",
        "NH4_sew": "NH4 Load [g/d]",
        # "PO4_sew": "PO4 Load [g/d]",
        # "COD_sol": "Dissolved COD Load [g/d]",
        # "X_TSS_sew": "Total Suspended Solids Load [g/d]",
        # "COD_part": "Particulate COD Load [g/d]",
    }
    labels_conc = {
        "H2O_sew": "Discharge [g/d]",
        "NH4_sew": "NH4 Concentration [g/g]",
        # "PO4_sew": "PO4 Concentration [g/g]",
        # "COD_sol": "Dissolved COD Concentration [g/g]",
        # "X_TSS_sew": "Total Suspended Solids Concentration [g/g]",
        # "COD_part": "Particulate COD Concentration [g/g]",
    }
    start_date = pd.Timestamp("2024-01-01")
    normal = pd.read_csv(
        rf"output_swmm\06-01_16-25_out_{catchment}_No_RTC.csv",
        decimal=",",
        delimiter=";",
        index_col=0,
    )
    normal["timestamp"] = start_date + pd.to_timedelta(
        normal.index.astype(float), unit="D"
    )
    normal.set_index("timestamp", inplace=True)
    normal.index = normal.index.round("15min")

    dry = pd.read_csv(
        rf"output_swmm\06-01_16-46_out_{catchment}_No_RTC_no_rain.csv",
        decimal=",",
        delimiter=";",
        index_col=0,
    )
    dry["timestamp"] = start_date + pd.to_timedelta(dry.index.astype(float), unit="D")
    dry.set_index("timestamp", inplace=True)
    dry.index = dry.index.round("15min")

    constant = pd.read_csv(
        rf"output_swmm\06-01_15-22_out_{catchment}_No_RTC_no_rain_constant.csv",
        decimal=",",
        delimiter=";",
        index_col=0,
    )
    constant["timestamp"] = start_date + pd.to_timedelta(
        constant.index.astype(float), unit="D"
    )
    constant.set_index("timestamp", inplace=True)
    constant.index = constant.index.round("15min")

    scenarios = {
        "Dry weather flow only": dry,
        "Constant dry weather flow only": constant,
        "Dry and wet weather flow": normal,
    }

    # Define your time window
    # start_date = pd.Timestamp("2024-05-16")
    # end_date = pd.Timestamp("2024-06-01")

    for key in list(dry.keys()):
        if not ("FD" in key or "Q_out" in key):
            # Filter by time window and store filtered versions
            filtered_scenarios = {
                key: df.loc[start_date:end_date] for key, df in scenarios.items()
            }

            # Determine subplot layout based on whether to include concentration
            include_conc = "H2O" not in key

            # Create figure and axes
            if include_conc:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
            else:
                fig, ax1 = plt.subplots(figsize=(14, 6))

            # --- Absolute values plot ---
            for label, df in filtered_scenarios.items():
                if key in df.columns:
                    ax1.plot(df.index, abs(df[key]), label=label)
                else:
                    print(f"Column '{key}' not found in scenario '{label}'")

            ax1.set_ylabel(labels[key])
            ax1.legend(loc=1)
            ax1.grid(True)

            # --- Concentration plot (only if applicable) ---
            if include_conc:
                for label, df in filtered_scenarios.items():
                    if key in df.columns and "H2O_sew" in df.columns:
                        ax2.plot(df.index, df[key] / df["H2O_sew"], label=label)
                    else:
                        print(
                            f"Column '{key}' or 'H2O_sew' not found in scenario '{label}'"
                        )

                ax2.set_ylabel(labels_conc[key])
                ax2.legend(loc=1)
                ax2.grid(True)
                ax1.set_xlabel("Date")
                ax2.set_xlabel("Date")
            else:
                ax1.set_xlabel("Date")

            # Final layout
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            for label in ax1.get_xticklabels():
                label.set_rotation(45)
            ax1.legend(loc=1, frameon=True, facecolor="white", framealpha=0.8)
            if include_conc:
                ax2.legend(loc=4, frameon=True, facecolor="white", framealpha=0.8)
                for label in ax2.get_xticklabels():
                    label.set_rotation(45)
            # plt.savefig(f"{key}")
            plt.show()

    from collections import defaultdict

    summary_stats = defaultdict(dict)
    # Iterate over all relevant pollutant keys
    for key in list(dry.keys()):
        if not ("FD" in key or "Q_out" in key):
            # Filter the time window
            filtered_scenarios = {
                label: df.loc[start_date:end_date] for label, df in scenarios.items()
            }

            for label, df in filtered_scenarios.items():
                if key not in df.columns:
                    continue  # Skip if key missing

                # Parse float just in case
                df_key = abs(df[[key]].astype(float))

                # 1. Load stats
                daily_avg_load = df_key.resample("D").mean()
                daily_max_load = df_key.resample("D").max()

                # 2. Concentration stats (if applicable)
                if "H2O" not in key:
                    flow = df["H2O_sew"].astype(float)
                    conc = df[key].astype(float) / flow.replace(0, pd.NA)

                    daily_avg_conc = conc.resample("D").mean()
                    daily_max_conc = conc.resample("D").max()

                    # Store average of daily stats
                    summary_stats[key][label] = {
                        "avg_daily_load": daily_avg_load.mean().values[0],
                        "max_daily_load": daily_max_load.mean().values[0],
                        "avg_daily_concentration": daily_avg_conc.mean(),
                        "max_daily_concentration": daily_max_conc.mean(),
                    }
                else:
                    summary_stats[key][label] = {
                        "avg_daily_load": daily_avg_load.mean().values[0],
                        "max_daily_load": daily_max_load.mean().values[0],
                        "avg_daily_concentration": None,
                        "max_daily_concentration": None,
                    }
    records = []
    for key, scenarios_dict in summary_stats.items():
        for scenario, metrics in scenarios_dict.items():
            record = {"Key": key, "Scenario": scenario, **metrics}
            records.append(record)
    summary_df_inflow = pd.DataFrame(records)

    labels = {
        ".WWTP2river.Outflow(rH2O)": "Discharge [g/d]",
        ".WWTP2river.Outflow(rBOD1)": "BOD1 Load [g/d]",
        ".WWTP2river.Outflow(rBOD1p)": "BOD1 Particulate Load [g/d]",
        ".WWTP2river.Outflow(rBOD2)": "BOD2 Load [g/d]",
        ".WWTP2river.Outflow(rBOD2p)": "BOD2 Particulate Load [g/d]",
        ".WWTP2river.Outflow(rBODs)": "BOD Solluable Load [g/d]",
        ".WWTP2river.Outflow(rNH4)": "NH4 Load [g/d]",
        ".WWTP2river.Outflow(rO2)": "O2 Load [g/d]",
    }

    labels_conc = {
        ".WWTP2river.Outflow(rH2O)": "Discharge [g/d]",
        ".WWTP2river.Outflow(rBOD1)": "BOD1 Concentration [g/g]",
        ".WWTP2river.Outflow(rBOD1p)": "BOD1 Particulate Concentration [g/g]",
        ".WWTP2river.Outflow(rBOD2)": "BOD2 Concentration [g/g]",
        ".WWTP2river.Outflow(rBOD2p)": "BOD2 Particulate Concentration [g/g]",
        ".WWTP2river.Outflow(rBODs)": "BOD Solluable Concentration [g/g]",
        ".WWTP2river.Outflow(rNH4)": "NH4 Concentration [g/g]",
        ".WWTP2river.Outflow(rO2)": "O2 Concentration [g/g]",
    }
    start_date = pd.Timestamp("2024-01-01")
    normal = pd.read_csv(
        f"data\WEST\Pollutant_no_RTC\comparison.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    normal["timestamp"] = start_date + pd.to_timedelta(
        normal.index.astype(float), unit="D"
    )
    normal.set_index("timestamp", inplace=True)
    normal.index = normal.index.round("15min")

    constant = pd.read_csv(
        f"data\WEST\Pollutant_no_RTC_no_rain_constant\comparison.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    constant["timestamp"] = start_date + pd.to_timedelta(
        constant.index.astype(float), unit="D"
    )
    constant.set_index("timestamp", inplace=True)
    constant.index = constant.index.round("15min")

    dry = pd.read_csv(
        f"data\WEST\Pollutant_no_RTC_no_rain\comparison.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    dry["timestamp"] = start_date + pd.to_timedelta(dry.index.astype(float), unit="D")
    dry.set_index("timestamp", inplace=True)
    dry.index = dry.index.round("15min")
    scenarios = {
        "Dry weather flow only": abs(dry.astype(float)),
        "Constant dry weather flow only": abs(constant.astype(float)),
        "Dry and wet weather flow": abs(normal.astype(float)),
    }
    # Define your time window
    start_date = pd.Timestamp("2024-05-16")
    end_date = pd.Timestamp("2024-06-01")

    for key in list(dry.keys()):
        if "WWTP2river" in key:
            # Filter by time window and store filtered versions
            filtered_scenarios = {
                key: df.loc[start_date:end_date] for key, df in scenarios.items()
            }

            include_conc = "H2O" not in key

            # Create figure and axes
            if include_conc:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
            else:
                fig, ax1 = plt.subplots(figsize=(14, 6))

            # --- Absolute values plot ---
            for label, df in filtered_scenarios.items():
                if key in df.columns:
                    ax1.plot(df.index, df[key].astype(float), label=label)
                else:
                    print(f"Column '{key}' not found in scenario '{label}'")

            ax1.set_xlabel("Date")
            ax1.set_ylabel(labels[key])
            ax1.grid(True)
            ax1.legend(loc=1, frameon=True, facecolor="white", framealpha=0.8)
            ax1.tick_params(axis="x", rotation=45)

            # --- Concentration plot (if applicable) ---
            if include_conc:
                for label, df in filtered_scenarios.items():
                    df = df.loc[start_date:end_date]
                    if key in df.columns and ".WWTP2river.Outflow(rH2O)" in df.columns:
                        ax2.plot(
                            df.index,
                            df[key].astype(float)
                            / df[".WWTP2river.Outflow(rH2O)"].astype(float),
                            label=label,
                        )
                    else:
                        print(
                            f"Column '{key}' or '.WWTP2river.Outflow(rH2O)' not found in '{label}'"
                        )

                ax2.set_xlabel("Date")
                ax2.set_ylabel(labels_conc[key])
                ax2.grid(True)
                ax2.legend(loc=1, frameon=True, facecolor="white", framealpha=0.8)
                ax2.tick_params(axis="x", rotation=45)

            # Final formatting
            fig.tight_layout()
            # fig.savefig(f"{key}.png")
            plt.show()

    from collections import defaultdict

    summary_stats = defaultdict(dict)
    # Iterate over all relevant pollutant keys
    for key in list(dry.keys()):
        if "WWTP2river" in key:
            # Filter the time window
            filtered_scenarios = {
                label: df.loc[start_date:end_date] for label, df in scenarios.items()
            }

            for label, df in filtered_scenarios.items():
                if key not in df.columns:
                    continue  # Skip if key missing

                # Parse float just in case
                df_key = abs(df[[key]].astype(float))

                # 1. Load stats
                daily_avg_load = df_key.resample("D").mean()
                daily_max_load = df_key.resample("D").max()

                # 2. Concentration stats (if applicable)
                if "H2O" not in key and ".WWTP2river.Outflow(rH2O)" in df.columns:
                    flow = df[".WWTP2river.Outflow(rH2O)"].astype(float)
                    conc = df[key].astype(float) / flow.replace(0, pd.NA)

                    daily_avg_conc = conc.resample("D").mean()
                    daily_max_conc = conc.resample("D").max()

                    # Store average of daily stats
                    summary_stats[key][label] = {
                        "avg_daily_load": daily_avg_load.mean().values[0],
                        "max_daily_load": daily_max_load.mean().values[0],
                        "avg_daily_concentration": daily_avg_conc.mean(),
                        "max_daily_concentration": daily_max_conc.mean(),
                    }
                else:
                    summary_stats[key][label] = {
                        "avg_daily_load": daily_avg_load.mean().values[0],
                        "max_daily_load": daily_max_load.mean().values[0],
                        "avg_daily_concentration": None,
                        "max_daily_concentration": None,
                    }
    records = []
    for key, scenarios_dict in summary_stats.items():
        for scenario, metrics in scenarios_dict.items():
            record = {"Key": key, "Scenario": scenario, **metrics}
            records.append(record)
    summary_df = pd.DataFrame(records)

    labels = {
        ".Well_35.Outflow(H2O_sew)": "Discharge [g/d]",
        ".Well_35.Outflow(NH4_sew)": "NH4 Load [g/d]",
        ".Well_35.Outflow(PO4_sew)": "PO4 Load [g/d]",
        ".Well_35.Outflow(COD_sol)": "Dissolved COD Load [g/d]",
        ".Well_35.Outflow(X_TSS_sew)": "Total Suspended Solids Load [g/d]",
        ".Well_35.Outflow(COD_part)": "Particulate COD Load [g/d]",
    }
    labels_conc = {
        ".Well_35.Outflow(H2O_sew)": "Discharge [g/d]",
        ".Well_35.Outflow(NH4_sew)": "NH4 Concentration [g/g]",
        ".Well_35.Outflow(PO4_sew)": "PO4 Concentration [g/g]",
        ".Well_35.Outflow(COD_sol)": "Dissolved COD Concentration [g/g]",
        ".Well_35.Outflow(X_TSS_sew)": "Total Suspended Solids Concentration [g/g]",
        ".Well_35.Outflow(COD_part)": "Particulate COD Concentration [g/g]",
    }
    start_date = pd.Timestamp("2024-01-01")
    normal = pd.read_csv(
        f"data\WEST\Pollutant_no_RTC\comparison.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    normal["timestamp"] = start_date + pd.to_timedelta(
        normal.index.astype(float), unit="D"
    )
    normal.set_index("timestamp", inplace=True)
    normal.index = normal.index.round("15min")

    constant = pd.read_csv(
        f"data\WEST\Pollutant_no_RTC_no_rain_constant\comparison.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    constant["timestamp"] = start_date + pd.to_timedelta(
        constant.index.astype(float), unit="D"
    )
    constant.set_index("timestamp", inplace=True)
    constant.index = constant.index.round("15min")

    dry = pd.read_csv(
        f"data\WEST\Pollutant_no_RTC_no_rain\comparison.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    dry["timestamp"] = start_date + pd.to_timedelta(dry.index.astype(float), unit="D")
    dry.set_index("timestamp", inplace=True)
    dry.index = dry.index.round("15min")
    scenarios = {
        "Dry weather flow only": abs(dry.astype(float)),
        "Constant dry weather flow only": abs(constant.astype(float)),
        "Dry and wet weather flow": abs(normal.astype(float)),
    }
    # Define your time window
    start_date = pd.Timestamp("2024-05-16")
    end_date = pd.Timestamp("2024-06-01")

    for key in list(dry.keys()):
        if "Well" in key:
            # Filter by time window and store filtered versions
            filtered_scenarios = {
                key: df.loc[start_date:end_date] for key, df in scenarios.items()
            }

            include_conc = "H2O" not in key

            # Create figure and axes
            if include_conc:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
            else:
                fig, ax1 = plt.subplots(figsize=(14, 6))

            # --- Absolute values plot ---
            for label, df in filtered_scenarios.items():
                if key in df.columns:
                    ax1.plot(df.index, df[key].astype(float), label=label)
                else:
                    print(f"Column '{key}' not found in scenario '{label}'")

            ax1.set_xlabel("Date")
            ax1.set_ylabel(labels[key])
            ax1.grid(True)
            ax1.legend(loc=1, frameon=True, facecolor="white", framealpha=0.8)
            ax1.tick_params(axis="x", rotation=45)

            # --- Concentration plot (if applicable) ---
            if include_conc:
                for label, df in filtered_scenarios.items():
                    df = df.loc[start_date:end_date]
                    if key in df.columns and ".Well_35.Outflow(H2O_sew)" in df.columns:
                        ax2.plot(
                            df.index,
                            df[key].astype(float)
                            / df[".Well_35.Outflow(H2O_sew)"].astype(float),
                            label=label,
                        )
                    else:
                        print(
                            f"Column '{key}' or '.Well_35.Outflow(H2O_sew)' not found in '{label}'"
                        )

                ax2.set_xlabel("Date")
                ax2.set_ylabel(labels_conc[key])
                ax2.grid(True)
                ax2.legend(loc=1, frameon=True, facecolor="white", framealpha=0.8)
                ax2.tick_params(axis="x", rotation=45)

            # Final formatting
            fig.tight_layout()
            # fig.savefig(f"{key}.png")
            plt.show()


def time_to_empty():
    #################################
    # Pump duration for different V and Pump flows smaller volume
    #################################

    # Constants
    P_max = 1 * 3600  # seconds
    V_max = 11000  # cubic meters

    # Pumping rate range (m³/s)
    Q_pump = np.linspace(0.01, 0.66, 200)

    # Time lines to plot (in seconds)
    time_hours = np.arange(2, 13, 2)  # hours
    time_lines = [t * 3600 for t in time_hours]

    # Plotting
    plt.figure(figsize=(10, 6))

    for t in time_lines:
        volume = Q_pump * t
        plt.plot(Q_pump, volume, label=f"{int(t / 3600)} h")

        # Place a label near the end of the line (on the left, since x-axis is flipped)
        x_text = Q_pump[0] - 0.03  # far left, due to inverted x-axis
        y_text = volume[0]

        # Check if the label goes above the upper limit of ylim

        # plt.text(x_text, y_text, f"{int(t / 3600)} h", va="center", ha="left", fontsize=9)

    # Optional: add max volume as horizontal line
    plt.axhline(11000, color="gray", linestyle="--", label="Max. volume Eindhoven")
    plt.axhline(7500, color="gray", linestyle="-.", label="Max. volume Riool Zuid")

    plt.xlabel("Pump Rate on top of DWF inflow (m³/s)")
    plt.ylabel("Total Volume Pumped (m³)")
    # plt.title("Total Volume Pumped vs Pump Rate for Different Durations")
    plt.ylim(top=V_max + 500)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()

    #####################################
    # Total volume stored analysis
    #####################################
    pattern_RZ = (
        pd.read_csv(
            rf"output_swmm\latest_out_RZ_DWF_only_out.csv",
            delimiter=";",
            decimal=",",
            index_col=0,
            parse_dates=True,
        )
        / 24
        / 3600
        * 5
        * 60
    )
    pattern_ES = (
        pd.read_csv(
            rf"output_swmm\latest_out_ES_DWF_only_out.csv",
            delimiter=";",
            decimal=",",
            index_col=0,
            parse_dates=True,
        )
        / 24
        / 3600
        * 5
        * 60
    )

    ES_inflow = pattern_ES.loc["2023-05-01":"2023-09-07", "H2O_sew"]
    ES_mean_inflow = ES_inflow.mean()
    RZ_inflow = pattern_RZ.loc["2023-05-01":"2023-09-07", "H2O_sew"]
    RZ_mean_inflow = RZ_inflow.mean()

    ES_storage = [0]
    RZ_storage = [0]
    for i in range(1, len(ES_inflow)):
        ES_delta = ES_inflow.iloc[i] - ES_mean_inflow
        ES_current_storage = max(ES_storage[-1] + ES_delta, 0)
        ES_storage.append(ES_current_storage)

        RZ_delta = RZ_inflow.iloc[i] - RZ_mean_inflow
        RZ_current_storage = max(RZ_storage[-1] + RZ_delta, 0)
        RZ_storage.append(RZ_current_storage)

    day_hour_formatter = mdates.DateFormatter(
        "%d %H"
    )  # Shows "day hour", e.g., "1 12" for May 1st, 12:00

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(ES_inflow.index, ES_storage, label="ES Storage")
    ax.plot(RZ_inflow.index, RZ_storage, label="RZ Storage")
    ax.set_xlabel("Day number and hour [dd hh]")
    ax.set_ylabel("Buffered storage Volume [m3]")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Set the major formatter to include both day and hour
    ax.xaxis.set_major_formatter(day_hour_formatter)

    # Set HourLocator to increase the number of x-ticks
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))  # Show every hour
    # ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # Show every 2 hours (comment/uncomment based on preference)

    # Adjust the limits so the x-axis doesn't go beyond the data range
    ax.set_xlim([RZ_inflow.index[0], RZ_inflow.index[-1]])

    plt.legend()

    # Rotate and align x labels
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.show()


def research_question_2():
    start_date = pd.Timestamp("2024-01-01")
    normal = pd.read_csv(
        rf"output_swmm\06-01_16-25_out_RZ_No_RTC.csv",
        decimal=",",
        delimiter=";",
        index_col=0,
    )
    normal["timestamp"] = start_date + pd.to_timedelta(
        normal.index.astype(float), unit="D"
    )
    normal.set_index("timestamp", inplace=True)
    normal.index = normal.index.round("5min")
    RTC = pd.read_csv(
        rf"output_swmm\06-01_13-14_out_RZ_RTC.csv",
        decimal=",",
        delimiter=";",
        index_col=0,
    )
    RTC["timestamp"] = start_date + pd.to_timedelta(RTC.index.astype(float), unit="D")
    RTC.set_index("timestamp", inplace=True)
    RTC.index = RTC.index.round("5min")

    start_date = pd.Timestamp("2024-05-09")
    end_date = pd.Timestamp("2024-05-15")

    normal_dry = normal.loc[start_date:end_date] / 1_000_000 / (24 * 3600)
    RTC_dry = RTC.loc[start_date:end_date] / 1_000_000 / (24 * 3600)

    plot(
        [normal_dry["H2O_sew"], RTC_dry["H2O_sew"]],
        [abs(normal_dry["H2O_sew"]), abs(RTC_dry["H2O_sew"])],
        ["Regular flow", "Controlled flow"],
        "Date and time",
        "Flow [m3/s]",
    )

    start_date = pd.Timestamp("2024-07-07")
    end_date = pd.Timestamp("2024-07-15")
    normal_wet = normal.loc[start_date:end_date] / 1_000_000 / (24 * 3600)
    RTC_wet = RTC.loc[start_date:end_date] / 1_000_000 / (24 * 3600)
    plot(
        [normal_wet["H2O_sew"], RTC_wet["H2O_sew"]],
        [abs(normal_wet["H2O_sew"]), abs(RTC_wet["H2O_sew"])],
        ["Regular flow", "Controlled flow"],
        "Date and time",
        "Flow [m3/s]",
    )

    normal_dry.std()
    RTC_dry.std()

    SWMM = sa.read_out_file("data\SWMM\model_jip.out").to_frame()
    SWMM = SWMM.loc[pd.Timestamp("2024-07-09") : pd.Timestamp("2024-07-15")]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot Flow on primary axis
    SWMM.link.P_riool_zuid_out.flow.plot(ax=ax1, label="Flow", color="tab:blue")
    ax1.set_ylabel("Flow [m³/s]")
    ax1.tick_params(axis="y")

    # Plot Volume on secondary axis
    sum(SWMM.link[pipe].volume for pipe in pipes).plot(
        ax=ax1, label="Volume", secondary_y=True, color="tab:orange"
    )
    ax1.right_ax.set_ylabel("Volume [m³]")
    ax1.right_ax.tick_params(axis="y")

    # Legend handling
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1.right_ax.get_legend_handles_labels()

    # Remove "(right)" from labels2
    labels2 = [label.replace(" (right)", "") for label in labels2]

    ax1.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=9)

    plt.xlabel("Date")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    ######################################################################
    start_date = pd.Timestamp("2024-01-01")
    normal = pd.read_csv(
        f"data\WEST\Pollutant_no_RTC\comparison.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    normal["timestamp"] = start_date + pd.to_timedelta(
        normal.index.astype(float), unit="D"
    )
    normal.set_index("timestamp", inplace=True)
    normal.index = normal.index.round("15min")
    RTC = pd.read_csv(
        f"data\WEST\Pollutant_RTC\comparison.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    RTC["timestamp"] = start_date + pd.to_timedelta(RTC.index.astype(float), unit="D")
    RTC.set_index("timestamp", inplace=True)
    RTC.index = RTC.index.round("15min")

    start_date = pd.Timestamp("2024-05-10")
    end_date = pd.Timestamp("2024-05-13")

    scenarios = {"normal": abs(normal.astype(float)), "RTC": abs(RTC.astype(float))}

    for key in list(normal.keys()):
        if "WWTP2river" in key:
            # Filter by time window and store filtered versions
            filtered_scenarios = {
                key: df.loc[start_date:end_date] for key, df in scenarios.items()
            }

            # Now plot all filtered scenarios together
            plt.figure(figsize=(14, 6))

            for label, df in filtered_scenarios.items():
                if key in df.columns:
                    plt.plot(df.index, df[key].astype(float), label=label)
                else:
                    print(f"Column '{key}' not found in scenario '{label}'")

            plt.xlabel("Date")
            plt.ylabel(key)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            if not "H2O" in key:
                plt.figure(figsize=(14, 6))
                for label, df in filtered_scenarios.items():
                    if key in df.columns:
                        plt.plot(
                            df.index,
                            df[key].astype(float)
                            / df[".WWTP2river.Outflow(rH2O)"].astype(float),
                            label=label,
                        )
                    else:
                        print(f"Column '{key}' not found in scenario '{label}'")

                plt.xlabel("Date")
                plt.ylabel(key + "Concentration")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
    from collections import defaultdict

    summary_stats = defaultdict(dict)
    # Iterate over all relevant pollutant keys
    for key in list(normal.keys()):
        if "WWTP2river" in key:
            # Filter the time window
            filtered_scenarios = {
                label: df.loc[start_date:end_date] for label, df in scenarios.items()
            }

            for label, df in filtered_scenarios.items():
                if key not in df.columns:
                    continue  # Skip if key missing

                # Parse float just in case
                df_key = df[[key]].astype(float)

                # 1. Load stats
                daily_avg_load = df_key.resample("D").mean()
                daily_max_load = df_key.resample("D").max()

                # 2. Concentration stats (if applicable)
                if "H2O" not in key and ".WWTP2river.Outflow(rH2O)" in df.columns:
                    flow = df[".WWTP2river.Outflow(rH2O)"].astype(float)
                    conc = df[key].astype(float) / flow.replace(0, pd.NA)

                    daily_avg_conc = conc.resample("D").mean()
                    daily_max_conc = conc.resample("D").max()

                    # Store average of daily stats
                    summary_stats[key][label] = {
                        "avg_daily_load": daily_avg_load.mean().values[0],
                        "max_daily_load": daily_max_load.mean().values[0],
                        "avg_daily_concentration": daily_avg_conc.mean(),
                        "max_daily_concentration": daily_max_conc.mean(),
                    }
                else:
                    summary_stats[key][label] = {
                        "avg_daily_load": daily_avg_load.mean().values[0],
                        "max_daily_load": daily_max_load.mean().values[0],
                        "avg_daily_concentration": None,
                        "max_daily_concentration": None,
                    }
    records = []
    for key, scenarios_dict in summary_stats.items():
        for scenario, metrics in scenarios_dict.items():
            record = {"Key": key, "Scenario": scenario, **metrics}
            records.append(record)
    summary_df = pd.DataFrame(records)


def research_question_3():
    start_date = pd.Timestamp("2024-01-01")
    normal = pd.read_csv(
        rf"output_swmm\06-01_16-25_out_RZ_No_RTC.csv",
        decimal=",",
        delimiter=";",
        index_col=0,
    )
    normal["timestamp"] = start_date + pd.to_timedelta(
        normal.index.astype(float), unit="D"
    )
    normal.set_index("timestamp", inplace=True)
    normal.index = normal.index.round("5min")
    RTC = pd.read_csv(
        rf"output_swmm\06-04_11-40_out_RZ_RTC.csv",
        decimal=",",
        delimiter=";",
        index_col=0,
    )
    RTC["timestamp"] = start_date + pd.to_timedelta(RTC.index.astype(float), unit="D")
    RTC.set_index("timestamp", inplace=True)
    RTC.index = RTC.index.round("5min")

    ENSEMBLE = pd.read_csv(
        rf"output_swmm\06-04_12-10_out_RZ_Ensemble_RTC.csv",
        decimal=",",
        delimiter=";",
        index_col=0,
    )
    ENSEMBLE["timestamp"] = start_date + pd.to_timedelta(
        ENSEMBLE.index.astype(float), unit="D"
    )
    ENSEMBLE.set_index("timestamp", inplace=True)
    ENSEMBLE.index = ENSEMBLE.index.round("5min")

    start_date = pd.Timestamp("2024-05-11")
    end_date = pd.Timestamp("2024-05-17")

    normal_dry = normal.loc[start_date:end_date] / 1_000_000 / (24 * 3600)
    RTC_dry = RTC.loc[start_date:end_date] / 1_000_000 / (24 * 3600)
    ENSEMBLE = ENSEMBLE.loc[start_date:end_date] / 1_000_000 / (24 * 3600)

    plot(
        [normal_dry["H2O_sew"], RTC_dry["H2O_sew"], ENSEMBLE["H2O_sew"]],
        [abs(normal_dry["H2O_sew"]), abs(RTC_dry["H2O_sew"]), abs(ENSEMBLE["H2O_sew"])],
        ["Regular flow", "Controlled flow", "Ensemble controlled flow"],
        "Date and time",
        "Flow [m3/s]",
    )

    states = pd.read_csv(
        rf"simulation_states_systems.csv", index_col=0, parse_dates=True
    )
    filtered_states = states.loc[start_date:end_date, ["RZ_RTC", "RZ_Ensemble_RTC"]]

    normal_dry.std()
    RTC_dry.std()

    SWMM = sa.read_out_file("data\SWMM\model_jip.out").to_frame()
    SWMM = SWMM.loc[pd.Timestamp("2024-07-08") : pd.Timestamp("2024-07-15")]
    ENSEMBLE_SWMM = sa.read_out_file("data\SWMM\model_jip.out").to_frame()
    ENSEMBLE_SWMM = ENSEMBLE_SWMM.loc[
        pd.Timestamp("2024-07-08") : pd.Timestamp("2024-07-15")
    ]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot Flow on primary axis
    SWMM.link.P_eindhoven_out.flow.plot(ax=ax1, label="Flow", color="tab:blue")
    ENSEMBLE_SWMM.link.P_eindhoven_out.flow.plot(ax=ax1, label="Flow", color="tab:blue")
    ax1.set_ylabel("Flow [m³/s]")
    ax1.tick_params(axis="y")

    # Plot Volume on secondary axis
    SWMM.node.pipe_ES.volume.plot(
        ax=ax1, label="Volume", secondary_y=True, color="tab:orange"
    )
    ENSEMBLE_SWMM.node.pipe_ES.volume.plot(
        ax=ax1, label="Volume", secondary_y=True, color="tab:orange"
    )
    ax1.right_ax.set_ylabel("Volume [m³]")
    ax1.right_ax.tick_params(axis="y")

    # Legend handling
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1.right_ax.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    plt.xlabel("Date")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def analyse_ensemble_flipper():
    start_date = pd.Timestamp("2024-01-01")
    RTC = pd.read_csv(
        rf"output_swmm\06-04_11-40_out_RZ_RTC.csv",
        decimal=",",
        delimiter=";",
        index_col=0,
    )
    RTC["timestamp"] = start_date + pd.to_timedelta(RTC.index.astype(float), unit="D")
    RTC.set_index("timestamp", inplace=True)
    RTC.index = RTC.index.round("5min")

    ENSEMBLE2 = pd.read_csv(
        rf"output_swmm\06-16_21-33_out_RZ_Ensemble_RTC_new_rain.csv",
        decimal=",",
        delimiter=";",
        index_col=0,
    )
    ENSEMBLE2["timestamp"] = start_date + pd.to_timedelta(
        ENSEMBLE2.index.astype(float), unit="D"
    )
    ENSEMBLE2.set_index("timestamp", inplace=True)
    ENSEMBLE2.index = ENSEMBLE2.index.round("5min")

    ENSEMBLE = pd.read_csv(
        rf"output_swmm\06-04_12-10_out_RZ_Ensemble_RTC.csv",
        decimal=",",
        delimiter=";",
        index_col=0,
    )
    ENSEMBLE["timestamp"] = start_date + pd.to_timedelta(
        ENSEMBLE.index.astype(float), unit="D"
    )
    ENSEMBLE.set_index("timestamp", inplace=True)
    ENSEMBLE.index = ENSEMBLE.index.round("5min")

    normal = pd.read_csv(
        rf"output_swmm\06-01_16-25_out_RZ_No_RTC.csv",
        decimal=",",
        delimiter=";",
        index_col=0,
    )
    normal["timestamp"] = start_date + pd.to_timedelta(
        normal.index.astype(float), unit="D"
    )
    normal.set_index("timestamp", inplace=True)
    normal.index = normal.index.round("5min")

    start_date = pd.Timestamp("2024-05-11")
    end_date = pd.Timestamp("2024-05-16")

    swmm = sa.read_out_file(rf"data\SWMM\model_jip_ENSEMBLE.out").to_frame()

    RTC_select = RTC.loc[start_date:end_date] / 1_000_000 / (24 * 3600)
    ENSEMBLE_select = ENSEMBLE.loc[start_date:end_date] / 1_000_000 / (24 * 3600)
    ENSEMBLE2_select = ENSEMBLE2.loc[start_date:end_date] / 1_000_000 / (24 * 3600)
    normal_select = normal.loc[start_date:end_date] / 1_000_000 / (24 * 3600)

    states = pd.read_csv(
        rf"simulation_states_systems.csv", index_col=0, parse_dates=True
    )
    filtered_states = states.loc[start_date:end_date, ["RZ_RTC", "RZ_Ensemble_RTC"]]

    # Collect all unique states
    rz_states = filtered_states["RZ_RTC"].unique()
    ens_states = filtered_states["RZ_Ensemble_RTC"].unique()

    # Assign different colormaps to each catchment
    rz_cmap = plt.cm.get_cmap("Blues", len(rz_states))
    ens_cmap = plt.cm.get_cmap("Oranges", len(ens_states))

    rz_colors = {state: rz_cmap(i) for i, state in enumerate(rz_states)}
    ens_colors = {state: ens_cmap(i) for i, state in enumerate(ens_states)}

    # Initialize one-time labels
    label_once = {f"RZ_{s}": True for s in rz_states}
    label_once.update({f"ENS_{s}": True for s in ens_states})

    precipitation = pd.read_csv(
        r"data\precipitation\csv_selected_area_euradclim\2024_5_min_precipitation_data.csv",
        index_col=0,
        parse_dates=True,
    )
    precipitation = precipitation.resample("h").sum()
    precipitation = precipitation.loc[start_date:end_date, ["GE", "RZ1", "RZ2"]]
    precipitation = precipitation.sum(axis=1)

    # Create main figure and primary axis
    fig, ax = plt.subplots(figsize=(12, 5))

    # Primary axis: flow plots
    ax.plot(
        RTC_select.index,
        abs(RTC_select["H2O_sew"].values),
        label="Controlled flow",
        zorder=3,
    )
    ax.plot(
        ENSEMBLE_select.index,
        abs(ENSEMBLE_select["H2O_sew"].values),
        label="Ensemble controlled flow",
        zorder=3,
    )
    ax.plot(
        ENSEMBLE2_select.index,
        abs(ENSEMBLE2_select["H2O_sew"].values),
        label="Ensemble2 controlled flow",
        zorder=3,
    )
    ax.plot(
        normal_select.index,
        abs(normal_select["H2O_sew"].values),
        label="Regular flow",
        zorder=3,
    )

    # Secondary axis: precipitation
    ax2 = ax.twinx()
    ax2.plot(
        precipitation.index,
        precipitation.values,
        color="tab:blue",
        linestyle="--",
        label="Precipitation [mm/h]",
        zorder=2,
    )
    ax2.set_ylabel("Precipitation [mm/h]", fontsize=10, color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    # Get y-limits for flow shading
    ymin, ymax = ax.get_ylim()

    # --- Shade RZ_RTC states (top half)
    prev_state = None
    start_time = None
    for time, state in filtered_states["RZ_RTC"].items():
        if state != prev_state:
            if prev_state is not None:
                ax.axvspan(
                    start_time,
                    time,
                    ymin=0.5,
                    ymax=1.0,
                    color=rz_colors[prev_state],
                    alpha=0.2,
                    label=f"RZ: {prev_state}" if label_once[f"RZ_{prev_state}"] else "",
                )
                label_once[f"RZ_{prev_state}"] = False
            start_time = time
            prev_state = state
    if prev_state is not None:
        ax.axvspan(
            start_time,
            filtered_states.index[-1],
            ymin=0.5,
            ymax=1.0,
            color=rz_colors[prev_state],
            alpha=0.2,
            label=f"RZ: {prev_state}" if label_once[f"RZ_{prev_state}"] else "",
        )

    # --- Shade RZ_Ensemble_RTC states (bottom half)
    prev_state = None
    start_time = None
    for time, state in filtered_states["RZ_Ensemble_RTC"].items():
        if state != prev_state:
            if prev_state is not None:
                ax.axvspan(
                    start_time,
                    time,
                    ymin=0.0,
                    ymax=0.5,
                    color=ens_colors[prev_state],
                    alpha=0.2,
                    label=(
                        f"Ensemble: {prev_state}"
                        if label_once[f"ENS_{prev_state}"]
                        else ""
                    ),
                )
                label_once[f"ENS_{prev_state}"] = False
            start_time = time
            prev_state = state
    if prev_state is not None:
        ax.axvspan(
            start_time,
            filtered_states.index[-1],
            ymin=0.0,
            ymax=0.5,
            color=ens_colors[prev_state],
            alpha=0.2,
            label=f"Ensemble: {prev_state}" if label_once[f"ENS_{prev_state}"] else "",
        )

    # Final formatting
    ax.set_xlabel("Date and Time", fontsize=10)
    ax.set_ylabel("Flow [m³/s]", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    from matplotlib.patches import Patch

    # Custom legend entries for state shading
    state_patches = []

    # # RZ states
    # for state, color in rz_colors.items():
    #     patch = Patch(facecolor=color, alpha=0.2, label=f"RZ: {state}")
    #     state_patches.append(patch)

    # # Ensemble states
    # for state, color in ens_colors.items():
    #     patch = Patch(facecolor=color, alpha=0.2, label=f"Ensemble: {state}")
    #     state_patches.append(patch)

    # # Combine all legends
    # lines1, labels1 = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()s

    # # Combine and add to legend
    # all_handles = lines1 + lines2 + state_patches
    # all_labels = labels1 + labels2 + [p.get_label() for p in state_patches]
    # ax.legend(all_handles, all_labels, fontsize=9, loc='upper left')

    plt.tight_layout()
    plt.show()


def research_2_3_combined():
    catchment = "ES"
    start_date = pd.Timestamp("2024-01-01")
    NORMAL = pd.read_csv(
        rf"output_swmm\06-01_16-25_out_{catchment}_No_RTC.csv",
        decimal=",",
        delimiter=";",
        index_col=0,
    )
    NORMAL["timestamp"] = start_date + pd.to_timedelta(
        NORMAL.index.astype(float), unit="D"
    )
    NORMAL.set_index("timestamp", inplace=True)
    NORMAL.index = NORMAL.index.round("5min")
    ENSEMBLE = pd.read_csv(
        rf"output_swmm\06-01_13-50_out_{catchment}_Ensemble_RTC.csv",
        decimal=",",
        delimiter=";",
        index_col=0,
    )
    ENSEMBLE["timestamp"] = start_date + pd.to_timedelta(
        ENSEMBLE.index.astype(float), unit="D"
    )
    ENSEMBLE.set_index("timestamp", inplace=True)
    ENSEMBLE.index = ENSEMBLE.index.round("5min")

    RTC = pd.read_csv(
        rf"output_swmm\06-01_13-14_out_{catchment}_RTC.csv",
        decimal=",",
        delimiter=";",
        index_col=0,
    )
    RTC["timestamp"] = start_date + pd.to_timedelta(RTC.index.astype(float), unit="D")
    RTC.set_index("timestamp", inplace=True)
    RTC.index = RTC.index.round("5min")

    IDEAL = pd.read_csv(
        rf"output_swmm\06-01_15-22_out_{catchment}_No_RTC_no_rain_constant.csv",
        decimal=",",
        delimiter=";",
        index_col=0,
    )
    IDEAL["timestamp"] = start_date + pd.to_timedelta(
        IDEAL.index.astype(float), unit="D"
    )
    IDEAL.set_index("timestamp", inplace=True)
    IDEAL.index = IDEAL.index.round("5min")

    start_date = pd.Timestamp("2024-07-07")
    end_date = pd.Timestamp("2024-07-15")
    normal_wet = NORMAL.loc[start_date:end_date] / 1_000_000 / (24 * 3600)
    RTC_wet = RTC.loc[start_date:end_date] / 1_000_000 / (24 * 3600)
    ENSEMBLE_wet = ENSEMBLE.loc[start_date:end_date] / 1_000_000 / (24 * 3600)
    # plot(
    #     [normal_wet["H2O_sew"], RTC_wet["H2O_sew"], ENSEMBLE_wet["H2O_sew"]],
    #     [
    #         abs(normal_wet["H2O_sew"]),
    #         abs(RTC_wet["H2O_sew"]),
    #         abs(ENSEMBLE_wet["H2O_sew"]),
    #     ],
    #     ["Regular flow", "Controlled flow", "Ensemble controlled flow"],
    #     "Date and time",
    #     "Flow [m3/s]",
    # )

    start_date = pd.Timestamp("2024-05-09")
    end_date = pd.Timestamp("2024-05-20")
    normal_dry = NORMAL.loc[start_date:end_date] / 1_000_000 / (24 * 3600)
    RTC_dry = RTC.loc[start_date:end_date] / 1_000_000 / (24 * 3600)
    ENSEMBLE_dry = ENSEMBLE.loc[start_date:end_date] / 1_000_000 / (24 * 3600)
    # plot(
    #     [normal_dry["H2O_sew"], RTC_dry["H2O_sew"], ENSEMBLE_dry["H2O_sew"]],
    #     [
    #         abs(normal_dry["H2O_sew"]),
    #         abs(RTC_dry["H2O_sew"]),
    #         abs(ENSEMBLE_dry["H2O_sew"]),
    #     ],
    #     ["Regular flow", "Controlled flow", "Ensemble controlled flow"],
    #     "Date and time",
    #     "Flow [m3/s]",
    # )

    ################# WWTP INFLUENT ##################
    labels = {
        "H2O_sew": "Discharge [g/d]",
        "NH4_sew": "NH4 Load [g/d]",
        "PO4_sew": "PO4 Load [g/d]",
        "COD_sol": "Dissolved COD Load [g/d]",
        "X_TSS_sew": "Total Suspended Solids Load [g/d]",
        "COD_part": "Particulate COD Load [g/d]",
    }
    labels_conc = {
        "H2O_sew": "Discharge [g/d]",
        "NH4_sew": "NH4 Concentration [g/g]",
        "PO4_sew": "PO4 Concentration [g/g]",
        "COD_sol": "Dissolved COD Concentration [g/g]",
        "X_TSS_sew": "Total Suspended Solids Concentration [g/g]",
        "COD_part": "Particulate COD Concentration [g/g]",
    }
    # Define your time window
    # DWF
    start_date = pd.Timestamp("2024-05-13")
    end_date = pd.Timestamp("2024-06-01")
    # WWF
    # start_date = pd.Timestamp("2024-05-01")
    # end_date = pd.Timestamp("2024-05-15")

    scenarios = {
        "RTC with ensembles": abs(ENSEMBLE.astype(float)),
        "No RTC": abs(NORMAL.astype(float)),
        "RTC": abs(RTC.astype(float)),
        "Constant flow": abs(IDEAL.astype(float)),
    }
    for key in list(NORMAL.keys()):
        if not ("FD" in key or "Q_out" in key):
            # Filter by time window and store filtered versions
            filtered_scenarios = {
                key: df.loc[start_date:end_date] for key, df in scenarios.items()
            }

            # Determine subplot layout based on whether to include concentration
            include_conc = "H2O" not in key

            # Create figure and axes
            if include_conc:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
            else:
                fig, ax1 = plt.subplots(figsize=(14, 6))

            # --- Absolute values plot ---
            for label, df in filtered_scenarios.items():
                if key in df.columns:
                    ax1.plot(df.index, abs(df[key]), label=label)
                else:
                    print(f"Column '{key}' not found in scenario '{label}'")

            ax1.set_ylabel(labels[key])
            ax1.legend(loc=1)
            ax1.grid(True)

            # --- Concentration plot (only if applicable) ---
            if include_conc:
                for label, df in filtered_scenarios.items():
                    if key in df.columns and "H2O_sew" in df.columns:
                        ax2.plot(df.index, df[key] / df["H2O_sew"], label=label)
                    else:
                        print(
                            f"Column '{key}' or 'H2O_sew' not found in scenario '{label}'"
                        )

                ax2.set_ylabel(labels_conc[key])
                ax2.legend(loc=1)
                ax2.grid(True)
                ax1.set_xlabel("Date")
                ax2.set_xlabel("Date")
            else:
                ax1.set_xlabel("Date")

            # Final layout
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            for label in ax1.get_xticklabels():
                label.set_rotation(45)
            ax1.legend(loc=1, frameon=True, facecolor="white", framealpha=0.8)
            if include_conc:
                ax2.legend(loc=4, frameon=True, facecolor="white", framealpha=0.8)
                for label in ax2.get_xticklabels():
                    label.set_rotation(45)
            plt.savefig(f"{catchment}{key}WWTPinfluentDWF")
            plt.show()

    from collections import defaultdict

    summary_stats = defaultdict(dict)
    # Iterate over all relevant pollutant keys
    for key in list(NORMAL.keys()):
        if not ("FD" in key or "Q_out" in key):
            # Filter the time window
            filtered_scenarios = {
                label: df.loc[start_date:end_date] for label, df in scenarios.items()
            }

            for label, df in filtered_scenarios.items():
                if key not in df.columns:
                    continue  # Skip if key missing

                # Parse float just in case
                df_key = abs(df[[key]].astype(float))

                # 1. Load stats
                daily_avg_load = df_key.resample("D").mean()
                daily_max_load = df_key.resample("D").max()

                # 2. Concentration stats (if applicable)
                if "H2O" not in key:
                    flow = df["H2O_sew"].astype(float)
                    conc = df[key].astype(float) / flow.replace(0, pd.NA)

                    daily_avg_conc = conc.resample("D").mean()
                    daily_max_conc = conc.resample("D").max()

                    # Store average of daily stats
                    summary_stats[key][label] = {
                        "avg_daily_load": daily_avg_load.mean().values[0],
                        "max_daily_load": daily_max_load.mean().values[0],
                        "avg_daily_concentration": daily_avg_conc.mean(),
                        "max_daily_concentration": daily_max_conc.mean(),
                    }
                else:
                    summary_stats[key][label] = {
                        "avg_daily_load": daily_avg_load.mean().values[0],
                        "max_daily_load": daily_max_load.mean().values[0],
                        "avg_daily_concentration": None,
                        "max_daily_concentration": None,
                    }
    records = []
    for key, scenarios_dict in summary_stats.items():
        for scenario, metrics in scenarios_dict.items():
            record = {"Key": key, "Scenario": scenario, **metrics}
            records.append(record)
    summary_df_inflow = pd.DataFrame(records)
    summary_df_inflow.to_csv(f"{catchment}WWTPinfluentDWF.csv")

    #################### WWTP EFFLUENT ################
    labels = {
        ".WWTP2river.Outflow(rH2O)": "Discharge [g/d]",
        ".WWTP2river.Outflow(rBOD1)": "BOD1 Load [g/d]",
        ".WWTP2river.Outflow(rBOD1p)": "BOD1 Particulate Load [g/d]",
        ".WWTP2river.Outflow(rBOD2)": "BOD2 Load [g/d]",
        ".WWTP2river.Outflow(rBOD2p)": "BOD2 Particulate Load [g/d]",
        ".WWTP2river.Outflow(rBODs)": "BOD Solluable Load [g/d]",
        ".WWTP2river.Outflow(rNH4)": "NH4 Load [g/d]",
        ".WWTP2river.Outflow(rO2)": "O2 Load [g/d]",
    }

    labels_conc = {
        ".WWTP2river.Outflow(rH2O)": "Discharge [g/d]",
        ".WWTP2river.Outflow(rBOD1)": "BOD1 Concentration [g/g]",
        ".WWTP2river.Outflow(rBOD1p)": "BOD1 Particulate Concentration [g/g]",
        ".WWTP2river.Outflow(rBOD2)": "BOD2 Concentration [g/g]",
        ".WWTP2river.Outflow(rBOD2p)": "BOD2 Particulate Concentration [g/g]",
        ".WWTP2river.Outflow(rBODs)": "BOD Solluable Concentration [g/g]",
        ".WWTP2river.Outflow(rNH4)": "NH4 Concentration [g/g]",
        ".WWTP2river.Outflow(rO2)": "O2 Concentration [g/g]",
    }
    start_date = pd.Timestamp("2024-01-01")
    NORMAL = pd.read_csv(
        f"data\WEST\Pollutant_no_RTC\comparison.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    NORMAL["timestamp"] = start_date + pd.to_timedelta(
        NORMAL.index.astype(float), unit="D"
    )
    NORMAL.set_index("timestamp", inplace=True)
    NORMAL.index = NORMAL.index.round("15min")

    RTC = pd.read_csv(
        f"data\WEST\Pollutant_RTC\comparison.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    RTC["timestamp"] = start_date + pd.to_timedelta(RTC.index.astype(float), unit="D")
    RTC.set_index("timestamp", inplace=True)
    RTC.index = RTC.index.round("15min")

    ENSEMBLE = pd.read_csv(
        f"data\WEST\Pollutant_RTC_ensemble\comparison.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    ENSEMBLE["timestamp"] = start_date + pd.to_timedelta(
        ENSEMBLE.index.astype(float), unit="D"
    )
    ENSEMBLE.set_index("timestamp", inplace=True)
    ENSEMBLE.index = ENSEMBLE.index.round("15min")

    IDEAL = pd.read_csv(
        f"data\WEST\Pollutant_no_RTC_no_rain_constant\comparison.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    IDEAL["timestamp"] = start_date + pd.to_timedelta(
        IDEAL.index.astype(float), unit="D"
    )
    IDEAL.set_index("timestamp", inplace=True)
    IDEAL.index = IDEAL.index.round("15min")

    scenarios = {
        "RTC with ensembles": abs(ENSEMBLE.astype(float)),
        "No RTC": abs(NORMAL.astype(float)),
        "RTC": abs(RTC.astype(float)),
        # "Constant flow": abs(IDEAL.astype(float))
    }
    # Define your time window
    start_date = pd.Timestamp("2024-05-10")
    end_date = pd.Timestamp("2024-06-01")

    for key in list(NORMAL.keys()):
        if "WWTP2river" in key:
            # Filter by time window and store filtered versions
            filtered_scenarios = {
                key: df.loc[start_date:end_date] for key, df in scenarios.items()
            }

            include_conc = "H2O" not in key

            if include_conc:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), sharex=False)
            else:
                fig, ax1 = plt.subplots(figsize=(14, 6))

            # --- Absolute values plot ---
            for label, df in filtered_scenarios.items():
                if key in df.columns:
                    ax1.plot(df.index, df[key].astype(float), label=label)
                else:
                    print(f"Column '{key}' not found in scenario '{label}'")

            ax1.set_xlabel("Date")
            ax1.set_ylabel(labels[key])
            ax1.grid(True)
            ax1.legend(loc=1, frameon=True, facecolor="white", framealpha=0.8)
            ax1.tick_params(axis="x", rotation=45)

            # --- Concentration plot (if applicable) ---
            if include_conc:
                for label, df in filtered_scenarios.items():
                    df = df.loc[start_date:end_date]
                    if key in df.columns and ".WWTP2river.Outflow(rH2O)" in df.columns:
                        ax2.plot(
                            df.index,
                            df[key].astype(float)
                            / df[".WWTP2river.Outflow(rH2O)"].astype(float),
                            label=label,
                        )
                    else:
                        print(
                            f"Column '{key}' or '.WWTP2river.Outflow(rH2O)' not found in '{label}'"
                        )

                ax2.set_xlabel("Date")
                ax2.set_ylabel(labels_conc[key])
                ax2.grid(True)
                ax2.legend(loc=1, frameon=True, facecolor="white", framealpha=0.8)
                ax2.tick_params(axis="x", rotation=45)

            # Final formatting
            fig.tight_layout()
            fig.savefig(f"{key}WWTPeffluentWWF.png")
            plt.show()
    from collections import defaultdict

    summary_stats = defaultdict(dict)
    # Iterate over all relevant pollutant keys
    for key in list(NORMAL.keys()):
        if "WWTP2river" in key:
            # Filter the time window
            filtered_scenarios = {
                label: df.loc[start_date:end_date] for label, df in scenarios.items()
            }

            for label, df in filtered_scenarios.items():
                if key not in df.columns:
                    continue  # Skip if key missing

                # Parse float just in case
                df_key = abs(df[[key]].astype(float))

                # 1. Load stats
                daily_avg_load = df_key.resample("D").mean()
                daily_max_load = df_key.resample("D").max()

                # 2. Concentration stats (if applicable)
                if "H2O" not in key and ".WWTP2river.Outflow(rH2O)" in df.columns:
                    flow = df[".WWTP2river.Outflow(rH2O)"].astype(float)
                    conc = df[key].astype(float) / flow.replace(0, pd.NA)

                    daily_avg_conc = conc.resample("D").mean()
                    daily_max_conc = conc.resample("D").max()

                    # Store average of daily stats
                    summary_stats[key][label] = {
                        "avg_daily_load": daily_avg_load.mean().values[0],
                        "max_daily_load": daily_max_load.mean().values[0],
                        "avg_daily_concentration": daily_avg_conc.mean(),
                        "max_daily_concentration": daily_max_conc.mean(),
                    }
                else:
                    summary_stats[key][label] = {
                        "avg_daily_load": daily_avg_load.mean().values[0],
                        "max_daily_load": daily_max_load.mean().values[0],
                        "avg_daily_concentration": None,
                        "max_daily_concentration": None,
                    }
    records = []
    for key, scenarios_dict in summary_stats.items():
        for scenario, metrics in scenarios_dict.items():
            record = {"Key": key, "Scenario": scenario, **metrics}
            records.append(record)
    summary_df = pd.DataFrame(records)
    summary_df.to_csv("WWTPEffluentWWF.csv")

    labels = {
        ".Well_35.Outflow(H2O_sew)": "Discharge [g/d]",
        ".Well_35.Outflow(NH4_sew)": "NH4 Load [g/d]",
        ".Well_35.Outflow(PO4_sew)": "PO4 Load [g/d]",
        ".Well_35.Outflow(COD_sol)": "Dissolved COD Load [g/d]",
        ".Well_35.Outflow(X_TSS_sew)": "Total Suspended Solids Load [g/d]",
        ".Well_35.Outflow(COD_part)": "Particulate COD Load [g/d]",
    }
    labels_conc = {
        ".Well_35.Outflow(H2O_sew)": "Discharge [g/d]",
        ".Well_35.Outflow(NH4_sew)": "NH4 Concentration [g/g]",
        ".Well_35.Outflow(PO4_sew)": "PO4 Concentration [g/g]",
        ".Well_35.Outflow(COD_sol)": "Dissolved COD Concentration [g/g]",
        ".Well_35.Outflow(X_TSS_sew)": "Total Suspended Solids Concentration [g/g]",
        ".Well_35.Outflow(COD_part)": "Particulate COD Concentration [g/g]",
    }
    start_date = pd.Timestamp("2024-01-01")
    NORMAL = pd.read_csv(
        f"data\WEST\Pollutant_no_RTC\comparison.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    NORMAL["timestamp"] = start_date + pd.to_timedelta(
        NORMAL.index.astype(float), unit="D"
    )
    NORMAL.set_index("timestamp", inplace=True)
    NORMAL.index = NORMAL.index.round("15min")

    RTC = pd.read_csv(
        f"data\WEST\Pollutant_RTC\comparison.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    RTC["timestamp"] = start_date + pd.to_timedelta(RTC.index.astype(float), unit="D")
    RTC.set_index("timestamp", inplace=True)
    RTC.index = RTC.index.round("15min")

    ENSEMBLE = pd.read_csv(
        f"data\WEST\Pollutant_RTC_ensemble\comparison.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    ENSEMBLE["timestamp"] = start_date + pd.to_timedelta(
        ENSEMBLE.index.astype(float), unit="D"
    )
    ENSEMBLE.set_index("timestamp", inplace=True)
    ENSEMBLE.index = ENSEMBLE.index.round("15min")

    IDEAL = pd.read_csv(
        f"data\WEST\Pollutant_no_RTC_no_rain_constant\comparison.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    IDEAL["timestamp"] = start_date + pd.to_timedelta(
        IDEAL.index.astype(float), unit="D"
    )
    IDEAL.set_index("timestamp", inplace=True)
    IDEAL.index = IDEAL.index.round("15min")

    scenarios = {
        "RTC with ensembles": abs(ENSEMBLE.astype(float)),
        "No RTC": abs(NORMAL.astype(float)),
        "RTC": abs(RTC.astype(float)),
        # "Constant flow": abs(IDEAL.astype(float))
    }
    # Define your time window
    start_date = pd.Timestamp("2024-05-10")
    end_date = pd.Timestamp("2024-06-01")

    for key in list(NORMAL.keys()):
        if "Well" in key:
            # Filter by time window and store filtered versions
            filtered_scenarios = {
                key: df.loc[start_date:end_date] for key, df in scenarios.items()
            }

            include_conc = "H2O" not in key

            if include_conc:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), sharex=False)
            else:
                fig, ax1 = plt.subplots(figsize=(14, 6))

            # --- Absolute values plot ---
            for label, df in filtered_scenarios.items():
                if key in df.columns:
                    ax1.plot(df.index, df[key].astype(float), label=label)
                else:
                    print(f"Column '{key}' not found in scenario '{label}'")

            ax1.set_xlabel("Date")
            ax1.set_ylabel(labels[key])
            ax1.grid(True)
            ax1.legend(loc=1, frameon=True, facecolor="white", framealpha=0.8)
            ax1.tick_params(axis="x", rotation=45)

            # --- Concentration plot (if applicable) ---
            if include_conc:
                for label, df in filtered_scenarios.items():
                    df = df.loc[start_date:end_date]
                    if key in df.columns and ".Well_35.Outflow(H2O_sew)" in df.columns:
                        ax2.plot(
                            df.index,
                            df[key].astype(float)
                            / df[".Well_35.Outflow(H2O_sew)"].astype(float),
                            label=label,
                        )
                    else:
                        print(
                            f"Column '{key}' or '.Well_35.Outflow(H2O_sew)' not found in '{label}'"
                        )

                ax2.set_xlabel("Date")
                ax2.set_ylabel(labels_conc[key])
                ax2.grid(True)
                ax2.legend(loc=1, frameon=True, facecolor="white", framealpha=0.8)
                ax2.tick_params(axis="x", rotation=45)

            # Final formatting
            fig.tight_layout()
            fig.savefig(f"{key}WWTPeffluentWWF.png")
            plt.show()


def OF_all(catchment="ES", start="2024-04-15", end="2045-10-15"):
    start_date = pd.Timestamp("2024-01-01")
    normal = pd.read_csv(
        rf"output_swmm\06-01_16-25_out_{catchment}_No_RTC.csv",
        decimal=",",
        delimiter=";",
        index_col=0,
    )
    normal["timestamp"] = start_date + pd.to_timedelta(
        normal.index.astype(float), unit="D"
    )
    normal.set_index("timestamp", inplace=True)
    normal.index = normal.index.round("15min")
    normal_swmm = sa.read_out_file(rf"data\SWMM\model_jip_no_rtc.out").to_frame()
    normal_west = pd.read_csv(
        f"data\WEST\Pollutant_no_RTC\comparison.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    normal_west["timestamp"] = start_date + pd.to_timedelta(
        normal_west.index.astype(float), unit="D"
    )
    normal_west.set_index("timestamp", inplace=True)
    normal_west.index = normal_west.index.round("15min")

    dry = pd.read_csv(
        rf"output_swmm\06-01_16-46_out_{catchment}_No_RTC_no_rain.csv",
        decimal=",",
        delimiter=";",
        index_col=0,
    )
    dry["timestamp"] = start_date + pd.to_timedelta(dry.index.astype(float), unit="D")
    dry.set_index("timestamp", inplace=True)
    dry.index = dry.index.round("15min")
    dry_swmm = sa.read_out_file(rf"data\SWMM\model_jip_no_rtc_no_rain.out").to_frame()
    dry_west = pd.read_csv(
        f"data\WEST\Pollutant_no_RTC_no_rain\comparison.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    dry_west["timestamp"] = start_date + pd.to_timedelta(
        dry_west.index.astype(float), unit="D"
    )
    dry_west.set_index("timestamp", inplace=True)
    dry_west.index = dry_west.index.round("15min")

    constant = pd.read_csv(
        rf"output_swmm\06-01_15-22_out_{catchment}_No_RTC_no_rain_constant.csv",
        decimal=",",
        delimiter=";",
        index_col=0,
    )
    constant["timestamp"] = start_date + pd.to_timedelta(
        constant.index.astype(float), unit="D"
    )
    constant.set_index("timestamp", inplace=True)
    constant.index = constant.index.round("15min")
    constant_swmm = sa.read_out_file(
        rf"data\SWMM\model_jip_no_rtc_no_rain_constant.out"
    ).to_frame()
    constant_west = pd.read_csv(
        f"data\WEST\Pollutant_no_RTC_no_rain_constant\comparison.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    constant_west["timestamp"] = start_date + pd.to_timedelta(
        constant_west.index.astype(float), unit="D"
    )
    constant_west.set_index("timestamp", inplace=True)
    constant_west.index = constant_west.index.round("15min")

    RTC = pd.read_csv(
        rf"output_swmm\06-04_11-40_out_{catchment}_RTC.csv",
        decimal=",",
        delimiter=";",
        index_col=0,
    )
    RTC["timestamp"] = start_date + pd.to_timedelta(RTC.index.astype(float), unit="D")
    RTC.set_index("timestamp", inplace=True)
    RTC.index = RTC.index.round("15min")
    RTC_swmm = sa.read_out_file(rf"data\SWMM\model_jip.out").to_frame()
    RTC_west = pd.read_csv(
        f"data\WEST\Pollutant_RTC\comparison.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    RTC_west["timestamp"] = start_date + pd.to_timedelta(
        RTC_west.index.astype(float), unit="D"
    )
    RTC_west.set_index("timestamp", inplace=True)
    RTC_west.index = RTC_west.index.round("15min")

    ENSEMBLE = pd.read_csv(
        rf"output_swmm\06-04_12-10_out_{catchment}_Ensemble_RTC.csv",
        decimal=",",
        delimiter=";",
        index_col=0,
    )
    ENSEMBLE["timestamp"] = start_date + pd.to_timedelta(
        ENSEMBLE.index.astype(float), unit="D"
    )
    ENSEMBLE.set_index("timestamp", inplace=True)
    ENSEMBLE.index = ENSEMBLE.index.round("15min")
    ENSEMBLE_swmm = sa.read_out_file(rf"data\SWMM\model_jip_ENSEMBLE.out").to_frame()
    ENSEMBLE_west = pd.read_csv(
        f"data\WEST\Pollutant_RTC_ensemble\comparison.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    ENSEMBLE_west["timestamp"] = start_date + pd.to_timedelta(
        ENSEMBLE_west.index.astype(float), unit="D"
    )
    ENSEMBLE_west.set_index("timestamp", inplace=True)
    ENSEMBLE_west.index = ENSEMBLE_west.index.round("15min")

    states = pd.read_csv(
        rf"simulation_states_systems.csv", index_col=0, parse_dates=True
    )

    scenarios = {
        # "Dry weather flow only": {
        #     "out": dry,
        #     "swmm": dry_swmm,
        #     "west": dry_west,
        # },
        # "Constant dry weather flow only": {
        #     "out": constant,
        #     "swmm": constant_swmm,
        #     "west": constant_west,
        # },
        "Dry and wet weather flow": {
            "out": normal,
            "swmm": normal_swmm,
            "west": normal_west,
        },
        "RTC": {
            "out": RTC,
            "swmm": RTC_swmm,
            "states": states[["ES_RTC", "RZ_RTC"]],
            "west": RTC_west,
        },
        "ENSEMBLE": {
            "out": ENSEMBLE,
            "swmm": ENSEMBLE_swmm,
            "states": states[["ES_Ensemble_RTC", "RZ_Ensemble_RTC"]],
            "west": ENSEMBLE_west,
        },
    }

    OF_values = {}

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    for k, i in scenarios.items():
        outflow = i["out"]
        swmm = i["swmm"]
        states = i.get("states")
        west = i["west"]

        outflow = outflow.loc[start_ts:end_ts]
        west = west.loc[start_ts:end_ts]
        if states is not None:
            states = states.loc[start_ts:end_ts]

        flow, FD, state, ideal_flow, loads, concs = get_values(
            outflow, catchment, states
        )
        effluent_loads, effluent_concs = get_west_values(west)
        if states is not None:
            FD_start_WWF, var_DWF_trans = objective_function_1(
                flow, FD, state, None, None
            )
        else:
            FD_start_WWF, var_DWF_trans = None, None
        CSO_vol = objective_function_2(swmm, catchment, start_ts, end_ts)
        steps_within_5_perc = objective_function_3(flow, ideal_flow, margin=1.05)
        steps_within_15_perc = objective_function_3(flow, ideal_flow, margin=1.15)
        pollutants = objective_function_4(loads, concs)
        pollutants_effluent = objective_function_4(effluent_loads, effluent_concs)

        OF_values[k] = {
            "FD": FD_start_WWF,  # float or None
            "DWF_var": var_DWF_trans,  # float or None
            "CSO": CSO_vol,  # float
            "5_perc_steps": steps_within_5_perc,  # float
            "15_perc_steps": steps_within_15_perc,  # float
            "pollutants": pollutants,  # dict
            "pollutants_effluent": pollutants_effluent,  # dict
        }

    df = pd.json_normalize(OF_values, sep=".")
    df = df.T.reset_index()
    df.columns = ["full_key", "value"]

    # Split the key into up to 4 parts: scenario, group, pollutant (possibly with dot), metric
    def custom_split(key):
        parts = key.split(".")
        scenario = parts[0]
        group = parts[1] if len(parts) > 1 else None
        # The pollutant could include dots, so join all until the last one
        pollutant = (
            ".".join(parts[2:-1])
            if len(parts) > 3
            else parts[2] if len(parts) > 2 else None
        )
        metric = parts[-1] if len(parts) > 2 else None
        return pd.Series([scenario, group, pollutant, metric])

    df[["scenario", "group", "pollutant", "metric"]] = df["full_key"].apply(
        custom_split
    )

    # Final clean DataFrame
    df = df[["scenario", "group", "pollutant", "metric", "value"]]

    return df


def get_values(output, catchment, states):
    if catchment == "ES":
        ideal = 0.663
    else:
        ideal = 0.5218

    flow = output["H2O_sew"] / 1e6 / 24 / 3600  # g/d to cms
    FD = output["FD"]

    if states is not None:
        key = next((k for k in states.keys() if catchment in k), None)
        state = states[key]
    else:
        state = None

    loads = (
        output[["NH4_sew", "PO4_sew", "COD_sol", "X_TSS_sew", "COD_part"]]
        / 24
        / 3600
        * 1000
    )  # g/d to mg/s

    concs = {}
    for key in ["NH4_sew", "PO4_sew", "COD_sol", "X_TSS_sew", "COD_part"]:
        concs[key] = (
            output[key].astype(float) / output["H2O_sew"].astype(float) * 1e6
        )  # from g/d, to g/g, to mg/L
    concs = pd.DataFrame(concs)

    return abs(flow), FD, state, ideal, abs(loads), abs(concs)


def get_west_values(west):
    effluent_keys = [
        key for key in west.keys() if (".effluent." in key) and ("y_Q" not in key)
    ]
    # all keys in g/m3 except for .Q which is m3/h
    flow = west[".effluent.y_Q"].astype(float) / 3600  # from m3/h to cm/s

    concs = west[effluent_keys].astype(float)  # g/m3 === mg/L

    loads = {}
    for key in effluent_keys:
        loads[key] = abs(west[key].astype(float) * flow * 1000)

    return loads, abs(concs)


def objective_function_1(outflow, FD, state, outflow_ideal, output):
    wanted_state_outflow = np.logical_or(
        state.values == "dwf", state.values == "transition"
    )
    wanted_state_FD = get_first_wwf_state(state)

    initial_wwf_FD = FD[wanted_state_FD]
    inital_wwf_FD_mean = np.mean(initial_wwf_FD)

    # mask = filted_high_inflows(output, outflow_ideal, outflow, wanted_state_outflow)
    filtered = outflow[wanted_state_outflow]

    return inital_wwf_FD_mean, filtered.var()


def get_first_wwf_state(state):
    # Identify the condition for 'wwf'
    wwf_condition = state.values == "wwf"

    # Create a mask for the first 'wwf' in a consecutive sequence
    first_wwf = np.zeros_like(wwf_condition, dtype=bool)

    # Mark the first occurrence of 'wwf' in the sequence
    for i in range(1, len(wwf_condition)):
        if wwf_condition[i] and not wwf_condition[i - 1]:
            first_wwf[i] = True

    # If the first element is 'wwf', mark it as True as well
    first_wwf[0] = wwf_condition[0]
    return first_wwf


def objective_function_2(output, location, start_ts, end_ts):
    swmm_csos = {
        "ES": "cso_ES_1",
        "RZ": [
            "cso_gb_136",
            "cso_Geldrop",
            "cso_gb127",
            "cso_RZ",
            "cso_AALST",
            "cso_c_123",
            "cso_c_122",
            "cso_c_119_1",
            "cso_c_119_2",
            "cso_c_119_3",
            "cso_c_112",
            "cso_c_99",
        ],
    }
    csos = swmm_csos[location]
    # Check if 'csos' is a list or a string and sum accordingly
    if isinstance(csos, list):  # For RZ
        total_inflow = sum(
            output.node[cso]["total_inflow"].loc[start_ts:end_ts].values.sum()
            for cso in csos
        )
    else:  # For ES (single CSO)
        total_inflow = (
            output.node[csos]["total_inflow"].loc[start_ts:end_ts].values.sum()
        )

    return total_inflow


def objective_function_3(flow, ideal_flow, margin=1.1):
    lower_bound = ideal_flow / margin
    upper_bound = ideal_flow * margin

    within_margin = (flow >= lower_bound) & (flow <= upper_bound)

    ratio_within = within_margin.sum() / len(flow)
    return 1 / ratio_within if ratio_within != 0 else float("inf")


def objective_function_4(loads, concs):
    summary_stats = {}

    for key in loads.keys():
        load = loads[key]
        conc = concs[key]
        daily_max_load = load.resample("D").max()
        daily_max_conc = conc.resample("D").max()

        # Store average of daily stats
        summary_stats[key] = {
            "avg_daily_load": load.mean(),
            "avg_max_daily_load": daily_max_load.mean(),
            "avg_concentration": conc.mean(),
            "avg_max_daily_concentration": daily_max_conc.mean(),
        }
    return summary_stats
