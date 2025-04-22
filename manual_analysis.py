import swmm_api as sa
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio

from sklearn.metrics import r2_score


def compare_models():
    df_west = pd.read_csv(
        f"data\WEST\Model_Dommel_Full\CSO_AND_INFLOW.out.txt",
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

    output = sa.SwmmOutput(
        rf"data\SWMM\model_jip_WEST_regen_geen_extra_storage_ts5.out"
    ).to_frame()

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

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_west.index,
            y=df_west[".pipe_ES.Q_Out"].astype(float),
            mode="lines",
            name=f"WEST Pipe ES Q-out",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_west.index,
            y=df_west[".pipe_RZ.Q_Out"].astype(float),
            mode="lines",
            name=f"WEST Pipe RZ Q-out",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=output.index,
            y=output.node.out_ES.total_inflow * 3600,
            mode="lines",
            name=f"SWMM pipe ES Q-out",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=output.index,
            y=output.node.out_RZ.total_inflow * 3600,
            mode="lines",
            name=f"SWMM pipe RZ Q-out",
        )
    )
    for key in swmm_csos.keys():
        swmm_values = 0
        for swmm_cso in swmm_csos[key]:
            swmm_values += output.node[swmm_cso].total_inflow.values * 3600

        west_values = df_west[west_csos[key][:]].astype(float).sum(axis=1)

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


def read_WEST_output():
    df_west = pd.read_csv(
        f"data\WEST\Model_Dommel_Full\westcompare.out.txt",
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
    fig = go.Figure()

    # Turn variables to m3/s if applicable
    #  ['.ES_out.Q_in', '.RZ_out.Q_in', '.ST_106.FillingDegreeIn',
    #        '.ST_106.Q_Out', '.c_106.Rainfall', '.c_106.comb.Out_1(Rain)',
    #        '.c_106.comb2.Q_i', '.c_106.dwf.Q_out',
    #        '.c_106.evaporation.Evaporation', '.c_106.evaporation.MeanEvaporation',
    #        '.c_106.runoff.FillingDepressionImp', '.c_106.runoff.RunoffImp',
    #        '.pipe_ES.FillingDegree', '.pipe_ES.FillingDegreeTank',
    #        '.pipe_ES.Q_Out', '.pipe_ES_tr.Q_Out', '.pipe_ES_tr.Q_i',
    #        '.pipe_ES_tr.Qrel', '.pipe_RZ.FillingDegree',
    #        '.pipe_RZ.FillingDegreeTank', '.pipe_RZ.Q_Out', '.pipe_RZ_tr.Q_Out',
    #        '.pipe_RZ_tr.Q_i', '.pipe_RZ_tr.Qrel']
    units = [
        3600,
        3600,
        1,
        3600,
        1,
        24,
        3600 * 24,
        3600 * 24,
        1,
        24,
        10,
        3600 * 24,
        1,
        1,
        3600,
        3600 * 24,
        3600 * 24,
        1,
        1,
        1,
        3600,
        3600 * 24,
        3600 * 24,
        1,
    ]
    # 3600,3600,1,3600,3600*24,1,24,3600*24,3600*24,1,24,1,3600*24,3600*24

    for key, unit in zip(df_west.keys(), units):
        fig.add_trace(
            go.Scatter(
                x=df_west.index,
                y=df_west[key].astype(float) / unit,
                mode="lines",
                name=f"WEST {key}",
            )
        )

    output = sa.SwmmOutput(
        rf"data\SWMM\model_jip_WEST_data_replicated_copy.out"
    ).to_frame()
    fig.add_trace(
        go.Scatter(
            x=output.node["out_ES"]["total_inflow"].index,
            y=output.node["out_ES"]["total_inflow"],
            mode="lines",
            name=f"SWMM ES out",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=output.node["out_RZ"]["total_inflow"].index,
            y=output.node["out_RZ"]["total_inflow"],
            mode="lines",
            name=f"SWMM RZ out",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=output.subcatchment["c_106_catchment"]["rainfall"].index,
            y=output.node.c_106_vr_storage_13.lateral_inflow,
            mode="lines",
            name=f"SWMM c_106 lateral inflow",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=output.subcatchment["c_106_catchment"]["rainfall"].index,
            y=output.link["P_c_106_d_13"]["flow"],
            mode="lines",
            name=f"SWMM pump c 106 flow",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=output.subcatchment["c_106_catchment"]["rainfall"].index,
            y=output.subcatchment["c_106_catchment"]["evaporation"],
            mode="lines",
            name=f"SWMM c 106 evaporation",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=output.subcatchment["c_106_catchment"]["rainfall"].index,
            y=output.subcatchment["c_106_catchment"]["rainfall"],
            mode="lines",
            name=f"SWMM c 106 rainfall",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=output.subcatchment["c_106_catchment"]["rainfall"].index,
            y=output.subcatchment["c_106_catchment"]["runoff"],
            mode="lines",
            name=f"SWMM c 106 runoff",
        )
    )

    AREA = 11.3 * (100**2)  # m2
    evap_mm_hr = output.subcatchment["c_106_catchment"]["evaporation"] / (24 * 12)
    rain_mm_hr = output.subcatchment["c_106_catchment"]["rainfall"] / 12
    runoff_mm_hr = (
        output.subcatchment["c_106_catchment"]["runoff"] / AREA * 3600 * 1000 / 12
    )

    storage = (rain_mm_hr).cumsum() - (evap_mm_hr + runoff_mm_hr).cumsum()
    storage = storage / 20

    fig.add_trace(
        go.Scatter(
            x=output.subcatchment["c_106_catchment"]["rainfall"].index,
            y=storage,
            mode="lines",
            name=f"SWMM c 106 calculated storage fraction of max",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=output.subcatchment["c_106_catchment"]["rainfall"].index,
            y=evap_mm_hr,
            mode="lines",
            name=f"SWMM c 106 evap mm",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=output.subcatchment["c_106_catchment"]["rainfall"].index,
            y=rain_mm_hr,
            mode="lines",
            name=f"SWMM c 106 rain mm",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=output.subcatchment["c_106_catchment"]["rainfall"].index,
            y=runoff_mm_hr,
            mode="lines",
            name=f"SWMM c 106 runoff mm",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=output.subcatchment["c_106_catchment"]["rainfall"].index,
            y=output.node["pipe_ES"].depth / 3,
            mode="lines",
            name=f"SWMM ES Filling degree",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=output.subcatchment["c_106_catchment"]["rainfall"].index,
            y=output.node["pre_ontvangstkelder"].depth / 29.01,
            mode="lines",
            name=f"SWMM RZ Filling degree",
        )
    )

    pio.show(fig, renderer="browser")

    pyo.plot(
        fig,
        filename=f"testing.html",
        auto_open=True,
    )


def compare_swmm_west_outflow():
    df_west = pd.read_csv(
        f"data\WEST\Model_Dommel_Full\ModelOutflowComparison.out.txt",
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
    west_ES_Q = df_west[".pipe_ES_tr.Q_Out"].astype(float) / 3600 / 24
    west_RZ_Q = df_west[".pipe_RZ_tr.Q_Out"].astype(float) / 3600 / 24

    output = sa.SwmmOutput(rf"data\SWMM\model_jip_WEST_data_replicated.out").to_frame()
    swmm_ES_Q = output.node.out_ES.total_inflow.resample("15T").mean()
    swmm_RZ_Q = output.node.out_RZ.total_inflow.resample("15T").mean()

    west_ES_Q_aligned = west_ES_Q.reindex(
        swmm_ES_Q.index, method="nearest", tolerance="10min"
    )
    west_RZ_Q_aligned = west_RZ_Q.reindex(
        swmm_RZ_Q.index, method="nearest", tolerance="10min"
    )

    def nash_sutcliff(observed, modelled):
        obs_mean = observed.mean()
        NSE = (
            1 - ((observed - modelled) ** 2).sum() / ((observed - obs_mean) ** 2).sum()
        )
        return NSE

    NSE_ES = nash_sutcliff(swmm_ES_Q, west_ES_Q_aligned)
    NSE_RZ = nash_sutcliff(swmm_RZ_Q, west_RZ_Q_aligned)


def compare_swmm_west_effluent():
    df_west = pd.read_csv(
        f"data\WEST\Model_Dommel_Full\ModelOutflowComparison.out.txt",
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

    df_west_swmm = pd.read_csv(
        f"data\WEST\SWMM_inputs\SWMMModelOutflowComparison.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    start_date = pd.Timestamp("2024-01-01")
    df_west_swmm["timestamp"] = start_date + pd.to_timedelta(
        df_west_swmm.index.astype(float), unit="D"
    )
    df_west_swmm.set_index("timestamp", inplace=True)

    # df_west = df_west[list(df_west_swmm.keys())]

    r2_values = {}
    r2_values2 = {}

    fig = go.Figure()
    for key in [
        ".WWTP2river.Outflow(rBOD1)",
        ".WWTP2river.Outflow(rH2O)",
        ".WWTP2river.Outflow(rNH4)",
        ".WWTP2river.Outflow(rO2)",
        ".fractionation.Inflow(NH4_sew)",
    ]:
        fig.add_trace(
            go.Scatter(
                x=df_west.index,
                y=df_west[key].astype(float),
                mode="lines",
                name=f"WEST {key}",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_west_swmm.index,
                y=df_west_swmm[key].astype(float),
                mode="lines",
                name=f"SWMM Input WEST {key}",
            )
        )

        r2_values[key] = r2_score(df_west[key].values, df_west_swmm[key].values)
    for swmm, west in zip(
        [".ES_out.Inflow(H2O_sew)", ".RZ_out.Inflow(H2O_sew)"],
        [".pipe_ES_tr.Q_Out", ".pipe_RZ_tr.Q_Out"],
    ):
        fig.add_trace(
            go.Scatter(
                x=df_west.index,
                y=df_west[west].astype(float) * 1e6,
                mode="lines",
                name=f"WEST {west}",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_west_swmm.index,
                y=df_west_swmm[swmm].astype(float),
                mode="lines",
                name=f"SWMM Input WEST {swmm}",
            )
        )
        r2_values2[west] = r2_score(df_west[west].values, df_west_swmm[swmm].values)

    output = sa.SwmmOutput(rf"data\SWMM\model_jip_WEST_data_replicated.out").to_frame()
    for outfall in ["out_RZ", "out_ES"]:
        outfall_timeseries = pd.DataFrame(output["node"][outfall]["total_inflow"])
        H2O_sew = (
            outfall_timeseries["total_inflow"] * 1e6 * (24 * 3600)
        )  # From CMS to g/d
        H2O_sew = H2O_sew.resample("15MIN").mean()
        fig.add_trace(
            go.Scatter(
                x=H2O_sew.index,
                y=H2O_sew,
                mode="lines",
                name=f"SWMM output flow {outfall}",
            )
        )

    pio.show(fig, renderer="browser")


def analyse_west_dwf():
    df_west = pd.read_csv(
        f"data\WEST\SWMM_inputs_DWF_2\CompareInflows.out.txt",
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
    fig = go.Figure()

    for key in df_west.keys():
        fig.add_trace(
            go.Scatter(
                x=df_west.index,
                y=df_west[key].astype(float),
                mode="lines",
                name=f"WEST {key}",
            )
        )

    pio.show(fig, renderer="browser")

    keys = list(df_west.keys())
    ES = [key for key in keys if "ES_out.Inflow" in key]
    df_ES = df_west[ES].astype(float)
    df_ES_divided = df_ES.div(df_ES[".ES_out.Inflow(H2O_sew)"], axis=0)
    RZ = [key for key in keys if "RZ_out.Inflow" in key]
    df_RZ = df_west[RZ].astype(float)
    df_RZ_divided = df_RZ.div(df_RZ[".RZ_out.Inflow(H2O_sew)"], axis=0)

    fig = go.Figure()

    for key in df_RZ_divided.keys():
        fig.add_trace(
            go.Scatter(
                x=df_RZ_divided.index,
                y=df_RZ_divided[key].astype(float),
                mode="lines",
                name=f"WEST {key}",
            )
        )

    pio.show(fig, renderer="browser")

    fig = go.Figure()

    for key in df_ES_divided.keys():
        fig.add_trace(
            go.Scatter(
                x=df_ES_divided.index,
                y=df_ES_divided[key].astype(float),
                mode="lines",
                name=f"WEST {key}",
            )
        )

    pio.show(fig, renderer="browser")


def check_cso_west():
    df_west = pd.read_csv(
        f"data\WEST\Model_Dommel_Full\Output1_UrbanSystem_CSO.out.txt",
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
        if any(substring in key for substring in ["Heeze", "Leende"])
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
    ]

    csos = [
        cso_ES_1_keys,
        cso_RZ_keys,
        cso_AALST_keys,
        cso_gb_136_keys,
        cso_Geldrop_keys,
    ]
    results = []
    df_west[".Waalre.Q_in"] = df_west[".Waalre.Q_in"].astype(float) / 24

    for cso_name, cso in zip(
        ["ES_1", "RZ", "AALST", "GB_136", "Geldrop"],
        csos + [".Waalre.Q_in", ".Aalst.Q_in"],
    ):
        temp_df = df_west[cso].copy()
        sum_df = pd.DataFrame(temp_df.astype(float).sum(axis=1), columns=["flow"])

        # Identify overflow events (where flow > 0)
        sum_df["overflow_event"] = (sum_df["flow"] > 0).astype(int)

        # Create groups for each overflow event
        sum_df["group"] = (sum_df["overflow_event"].diff(1) == 1).cumsum()

        # Filter only overflow periods
        overflow_groups = sum_df[sum_df["flow"] > 0].groupby("group")

        # Collect overflow data
        for group, group_df in overflow_groups:
            num_overflows = len(overflow_groups)
            volume = group_df["flow"].sum() / 6  # Divide by 6 as in your code
            avg_timestamp = group_df.index.mean()

            # Append data to results
            results.append(
                {
                    "CSO": cso_name,
                    "Overflow Number": group,
                    "Volume (m³)": np.round(volume),
                    "Average Timestamp": avg_timestamp,
                }
            )

    # Create a DataFrame from the results
    overflow_summary = pd.DataFrame(results)

    # Save to CSV
    overflow_summary.to_csv("overflow_summary.csv", index=False)

    print(overflow_summary.head())


def check_cso_swmm():
    output = sa.SwmmOutput(
        rf"data\SWMM\model_jip_WEST_data_replicated_copy.out"
    ).to_frame()
    csos_swmm = [
        ["cso_ES_1"],
        ["cso_RZ"],
        [
            "cso_AALST",
            "cso_c_123",
            "cso_c_99",
            "cso_c_112",
            "cso_c_119",
            "cso_c_122",
        ],
        ["cso_gb_136"],
        ["cso_Geldrop"],
    ]

    results = []

    for cso_name, cso in zip(["ES_1", "RZ", "AALST", "GB_136", "Geldrop"], csos_swmm):
        values = 0
        for sub_cso in cso:
            values += output.node[sub_cso].total_inflow.values

        sum_df = pd.DataFrame(
            values, index=output.index, columns=["total_inflow"]
        )  # Identify overflow events (where flow > 0)
        sum_df["overflow_event"] = (sum_df["total_inflow"] > 0).astype(int)

        # Create groups for each overflow event
        sum_df["group"] = (sum_df["overflow_event"].diff(1) == 1).cumsum()

        # Filter only overflow periods
        overflow_groups = sum_df[sum_df["total_inflow"] > 0].groupby("group")

        # Collect overflow data
        for group, group_df in overflow_groups:
            num_overflows = len(overflow_groups)
            volume = (
                group_df["total_inflow"].sum() * 15 * 60
            )  # Divide by 6 as in your code
            avg_timestamp = group_df.index.mean()

            # Append data to results
            results.append(
                {
                    "CSO": cso_name,
                    "Overflow Number": group,
                    "Volume (m³)": np.round(volume),
                    "Average Timestamp": avg_timestamp,
                }
            )

    # Create a DataFrame from the results
    overflow_summary = pd.DataFrame(results)

    # Save to CSV
    overflow_summary.to_csv("overflow_summary_swmm.csv", index=False)

    print(overflow_summary.head())


import plotly.express as px


def compare_cso_overflows():
    # Load overflow summaries
    west_df = pd.read_csv("overflow_summary.csv", parse_dates=["Average Timestamp"])
    swmm_df = pd.read_csv(
        "overflow_summary_swmm.csv", parse_dates=["Average Timestamp"]
    )

    # Add model identifiers
    west_df["Model"] = "WEST"
    swmm_df["Model"] = "SWMM"

    # Combine the data
    combined_df = pd.concat([west_df, swmm_df], ignore_index=True)

    # Create the bar plot
    fig = px.bar(
        combined_df,
        x="Average Timestamp",
        y="Volume (m³)",
        color="Model",
        barmode="group",
        facet_row="CSO",
        title="CSO Overflow Comparison: WEST vs. SWMM",
        labels={"Volume (m³)": "Overflow Volume (m³)", "Average Timestamp": "Date"},
        height=1500,
        width=1400,
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Overflow Volume (m³)",
        legend_title="Model",
        bargap=0.1,  # Reducing gap between bars for better visibility
    )

    pio.show(fig, renderer="browser")


def plot_west_out():
    project = "SWMM_input_constant_DWF"
    df_west = pd.read_csv(
        f"data\WEST\{project}\WWTP_input.out.txt",
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
    fig = go.Figure()

    for key in df_west.keys():
        fig.add_trace(
            go.Scatter(
                x=df_west.index,
                y=df_west[key].astype(float),
                mode="lines",
                name=f"WEST {key}",
            )
        )

    pio.show(fig, renderer="browser")


def plot_wwtp_input_vs_output():
    projects = [
        "SWMM_input_constant_DWF",
        "SWMM_inputs_dwf_and_precipitation",
        "SWMM_inputs_dwf_only",
    ]

    for project in projects:
        print(project)
        fig = go.Figure()
        wwtp_input = pd.read_csv(
            f"data\WEST\{project}\WWTP_input.out.txt",
            delimiter="\t",
            header=0,
            index_col=0,
            low_memory=False,
        ).iloc[1:, :]
        start_date = pd.Timestamp("2024-01-01")
        wwtp_input["timestamp"] = start_date + pd.to_timedelta(
            wwtp_input.index.astype(float), unit="D"
        )
        wwtp_input.set_index("timestamp", inplace=True)

        for key in wwtp_input.keys():
            fig.add_trace(
                go.Scatter(
                    x=wwtp_input.index,
                    y=wwtp_input[key].astype(float),
                    mode="lines",
                    name=f"WEST {key}",
                )
            )

        wwtp_output = pd.read_csv(
            f"data\WEST\{project}\WWTP_output.out.txt",
            delimiter="\t",
            header=0,
            index_col=0,
            low_memory=False,
        ).iloc[1:, :]
        start_date = pd.Timestamp("2024-01-01")
        wwtp_output["timestamp"] = start_date + pd.to_timedelta(
            wwtp_output.index.astype(float), unit="D"
        )
        wwtp_output.set_index("timestamp", inplace=True)

        for key in wwtp_output.keys():
            fig.add_trace(
                go.Scatter(
                    x=wwtp_output.index,
                    y=wwtp_output[key].astype(float),
                    mode="lines",
                    name=f"WEST {key}",
                )
            )
            print(
                f'{key}: {wwtp_output.loc["2024-03-25":"2024-04-08", key].astype(float).sum()}'
            )

        print("\n")
        fig.update_layout(title_text=f"{project}")
        pio.show(fig, renderer="browser")

        pyo.plot(
            fig,
            filename=f"wwtp_comparison_{project}.html",
            auto_open=True,
        )


def compare_inflow():
    project = "SWMM_input_constant_DWF"
    df_west = pd.read_csv(
        rf"data\WEST\SWMM_inputs_dwf_only\NHcheck.out.txt",
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
    fig = go.Figure()

    for key in df_west.keys():
        fig.add_trace(
            go.Scatter(
                x=df_west.index,
                y=df_west[key].astype(float),
                mode="lines",
                name=f"WEST {key}",
            )
        )

    pio.show(fig, renderer="browser")
