import swmm_api as sa
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio
from plotly.subplots import make_subplots

from sklearn.metrics import r2_score


def check_load():
    df_west = pd.read_csv(
        r"output_swmm\latest_out_ES_out.csv",
        # rf'data\WEST\WEST_modelRepository\Model_Dommel_Full\wwtp_control.out.txt',
        delimiter=";",
        decimal=",",
        index_col=0,
        parse_dates=True,
    )

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

    df_west = pd.read_csv(
        r"output_swmm\06-04_11-40_out_ES_RTC.txt",
        # rf'data\WEST\WEST_modelRepository\Model_Dommel_Full\wwtp_control.out.txt',
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

    for key in df_west.keys():
        fig.add_trace(
            go.Scatter(
                x=df_west.index,
                y=df_west[key].astype(float),
                mode="lines",
                name=f"RTC base {key}",
            )
        )

    df_west = pd.read_csv(
        r"output_swmm\06-01_16-25_out_ES_No_RTC.txt",
        # rf'data\WEST\WEST_modelRepository\Model_Dommel_Full\wwtp_control.out.txt',
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

    for key in df_west.keys():
        fig.add_trace(
            go.Scatter(
                x=df_west.index,
                y=df_west[key].astype(float),
                mode="lines",
                name=f"NO RTC base {key}",
            )
        )

    pio.show(fig, renderer="browser")


def check_any_west_results():
    df_west = pd.read_csv(
        f"data\WEST\WEST_modelRepository\Model_Dommel_Full\wwtp_control.out.txt",
        # rf'data\WEST\WEST_modelRepository\Model_Dommel_Full\wwtp_control.out.txt',
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
    # df = pd.read_csv(
    #     rf"output_swmm\05-15_13-14_out_ES_No_RTC.csv",
    #     index_col=0,
    #     delimiter=";",
    #     decimal=",",
    # )
    # df["timestamp"] = start_date + pd.to_timedelta(df.index.astype(float), unit="D")
    # df.set_index("timestamp", inplace=True)

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

    # fig.add_trace(
    #     go.Scatter(
    #         x=df.index,
    #         y=df.FD,
    #         mode="lines",
    #         name="Swmm FD",
    #     )
    # )

    pio.show(fig, renderer="browser")


def compare_csos():
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
    df_west = df_west.loc["2024-04-15":"2024-10-15"]

    output = sa.SwmmOutput(rf"data\SWMM\model_jip_WEST_regen.out").to_frame()

    swmm_total = sum(
        output.node[cso].total_inflow * 3600
        for cso in [
            "cso_AALST",
            "cso_c_123",
            "cso_c_122",
            "cso_c_119_1",
            "cso_c_119_2",
            "cso_c_119_3",
            "cso_c_112",
            "cso_c_99",
        ]
    )

    swmm_series = pd.Series(swmm_total.values, index=output.index)
    swmm_resampled = swmm_series.resample("15min").mean()

    Q_keys = df_west.keys()
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

    west_series = df_west[cso_AALST_keys].astype(float).sum(axis=1)
    west_resampled = west_series.resample("15min").mean()

    def group_boolean_events(series: pd.Series, min_duration="15min"):
        """Group consecutive True values into event groups."""
        series = series.astype(bool)
        # Identify start of events
        event_id = (series != series.shift()).cumsum()
        grouped = series[series].groupby(event_id)

        events = []
        for _, group in grouped:
            start = group.index[0]
            end = group.index[-1]
            if (end - start) >= pd.to_timedelta(min_duration):
                events.append((start, end))
        return events

    cso_threshold = 0.01

    # Already resampled
    west_events = west_resampled > cso_threshold
    swmm_events = swmm_resampled > cso_threshold

    # Group WEST CSO events
    # Group events into start/end windows
    west_event_windows = group_boolean_events(west_events)
    swmm_event_windows = group_boolean_events(swmm_events)

    def overlaps(west_start, west_end, swmm_windows):
        for swmm_start, swmm_end in swmm_windows:
            # Check for overlap
            if west_end >= swmm_start and swmm_end >= west_start:
                return True
        return False

    rainfall_resampled = output.subcatchment.c_119_catchment.rainfall.resample(
        "15min"
    ).mean()

    from datetime import timedelta

    buffer = timedelta(hours=6)
    missed_events = []

    for start, end in west_event_windows:
        # Check for overlap between this WEST event and any SWMM event
        if not overlaps(start, end, swmm_event_windows):

            # RAINFALL: Get ±6h window around the WEST event
            rainfall_window = rainfall_resampled.loc[start - buffer : end + buffer]

            # CSO volumes (sum of flows × 15min timestep = m³/h × h)
            dt_hours = 15 / 60  # 15-minute step in hours

            total_west_volume = west_resampled.loc[start:end].sum() * dt_hours
            total_swmm_volume = swmm_resampled.loc[start:end].sum() * dt_hours
            total_rainfall_mm = rainfall_window.sum() * dt_hours

            missed_events.append(
                {
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "max_rainfall_±6h": rainfall_window.max(),
                    "mean_rainfall_±6h": rainfall_window.mean(),
                    "total_rainfall_±6h": total_rainfall_mm,
                    "total_west_cso_m3": total_west_volume,
                    "total_swmm_cso_m3": total_swmm_volume,
                }
            )

    missed_df = pd.DataFrame(missed_events).round(2)

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=swmm_resampled.index,
            y=swmm_resampled,
            mode="lines",
            name=f"SWMM cso flow RZ",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=west_resampled.index,
            y=west_resampled.values,
            mode="lines",
            name=f"WEST cso flow  RZ",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=output.index,
            y=output.subcatchment.c_119_catchment.rainfall.values,
            mode="lines",
            name="Valkenswaard precipitation",
            line=dict(color="green"),
            opacity=0.8,
        ),
        secondary_y=True,
    )
    # Update layout
    fig.update_layout(
        title_text="Total CSO flow per catchment comparison",
        xaxis_title="Date",
        yaxis_title="Flow [m3/h]",
        legend_title="",
        bargap=0.1,
    )

    # Set y-axis titles
    fig.update_yaxes(title_text="Flow [m3/h]", secondary_y=False)
    fig.update_yaxes(title_text="Precipitation [mm/h]", secondary_y=True)

    # Show figure
    pio.show(fig, renderer="browser")


def compare_models():
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
    df_west = df_west.loc["2024-04-15":"2025-10-15"]

    output = sa.SwmmOutput(rf"data\SWMM\model_jip_WEST_regen.out").to_frame()

    swmm_resampled = pd.Series(output, index=output.index).resample("15min").mean()
    west_resampled = df_west.resample("15min").mean()

    swmm_csos = {
        # "ES": ["cso_ES_1"],
        # "GB": ["cso_gb_136"],
        # "GE": ["cso_Geldrop", "cso_gb127"],
        # "TL": ["cso_RZ"],
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
        "RZ": cso_AALST_keys,
    }

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add SWMM and WEST CSO flows (primary y-axis)
    for key in swmm_csos.keys():
        swmm_values = 0
        for swmm_cso in swmm_csos[key]:
            swmm_values += (
                output.node[swmm_cso].total_inflow.values * 3600
            )  # Convert to m3/h

        west_values = df_west[west_csos[key][:]].astype(float).sum(axis=1)

        fig.add_trace(
            go.Scatter(
                x=output.index,
                y=swmm_values,
                mode="lines",
                name=f"SWMM cso flow {key}",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=west_values.index,
                y=west_values.values,
                mode="lines",
                name=f"WEST cso flow {key}",
            ),
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=output.index,
            y=output.subcatchment.c_119_catchment.rainfall.values,
            mode="lines",
            name="Valkenswaard precipitation",
            line=dict(color="green"),
            opacity=0.8,
        ),
        secondary_y=True,
    )
    # Update layout
    fig.update_layout(
        title_text="Total CSO flow per catchment comparison",
        xaxis_title="Date",
        yaxis_title="Flow [m3/h]",
        legend_title="",
        bargap=0.1,
    )

    # Set y-axis titles
    fig.update_yaxes(title_text="Flow [m3/h]", secondary_y=False)
    fig.update_yaxes(title_text="Precipitation [mm/h]", secondary_y=True)

    # Show figure
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

    output = sa.SwmmOutput(rf"data\SWMM\model_jip_WEST_regen.out").to_frame()
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


def compare_NHflow():
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

    df_west2 = pd.read_csv(
        rf"data\WEST\SWMM_input_constant_DWF\NHcheck.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    start_date = pd.Timestamp("2024-01-01")
    df_west2["timestamp"] = start_date + pd.to_timedelta(
        df_west2.index.astype(float), unit="D"
    )
    df_west2.set_index("timestamp", inplace=True)
    df_west = df_west.loc["2024-07-01":"2024-07-31"]
    df_west2 = df_west2.loc["2024-07-01":"2024-07-31"]
    fig = go.Figure()

    for key in df_west2.keys():
        fig.add_trace(
            go.Scatter(
                x=df_west.index,
                y=df_west[key].astype(float),
                mode="lines",
                name=f"Normal {key}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_west2.index,
                y=df_west2[key].astype(float),
                mode="lines",
                name=f"Constant {key}",
            )
        )

    pio.show(fig, renderer="browser")


def checkNHDose():
    df_west = pd.read_csv(
        rf"data\WEST\SWMM_inputs_dwf_only\C_Dose_check.1.out.txt",
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

    df_west2 = pd.read_csv(
        rf"data\WEST\C_dosed_constant_DWF\C_Dose_check.1.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    start_date = pd.Timestamp("2024-01-01")
    df_west2["timestamp"] = start_date + pd.to_timedelta(
        df_west2.index.astype(float), unit="D"
    )
    df_west2.set_index("timestamp", inplace=True)

    df_west3 = pd.read_csv(
        rf"data\WEST\SWMM_input_constant_DWF\C_Dose_check.1.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    start_date = pd.Timestamp("2024-01-01")
    df_west3["timestamp"] = start_date + pd.to_timedelta(
        df_west3.index.astype(float), unit="D"
    )
    df_west3.set_index("timestamp", inplace=True)

    df_west = df_west.loc["2024-07-01":"2024-07-31"]
    df_west2 = df_west2.loc["2024-07-01":"2024-07-31"]
    df_west3 = df_west3.loc["2024-07-01":"2024-07-31"]

    fig = go.Figure()

    for key in df_west2.keys():
        fig.add_trace(
            go.Scatter(
                x=df_west.index,
                y=abs(df_west[key].astype(float)),
                mode="lines",
                name=f"Normal {key}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_west2.index,
                y=abs(df_west2[key].astype(float)),
                mode="lines",
                name=f"Constant Dosed {key}",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_west3.index,
                y=abs(df_west3[key].astype(float)),
                mode="lines",
                name=f"Constant {key}",
            )
        )

    pio.show(fig, renderer="browser")


def compare_concentrations_BASE():
    ES_conc_buffered = pd.read_csv(
        rf"effluent_concentration\testing_results\RZ_concentrations.csv",
        index_col=0,
        parse_dates=True,
    )
    RZ_conc_buffered = pd.read_csv(
        rf"effluent_concentration\testing_results\ES_concentrations.csv",
        index_col=0,
        parse_dates=True,
    )
    ES_conc = pd.read_csv(rf"effluent_concentration\ES.Effluent.csv", index_col=0)
    RZ_conc = pd.read_csv(rf"effluent_concentration\RZ.Effluent.csv", index_col=0)

    df_west = pd.read_csv(
        rf"data\WEST\SWMM_inputs_dwf_and_precipitation\concentration_check.out.txt",
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
                y=abs(df_west[key].astype(float)),
                mode="lines",
                name=f"West {key}",
            )
        )

    for key in ["COD_part", "COD_sol", "X_TSS_sew", "NH4_sew", "PO4_sew"]:
        fig.add_trace(
            go.Scatter(
                x=ES_conc_buffered.index,
                y=ES_conc_buffered[key].astype(float),
                mode="lines",
                name=f"ES {key} BUFFERED",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=RZ_conc_buffered.index,
                y=RZ_conc_buffered[key].astype(float),
                mode="lines",
                name=f"RZ {key} BUFFERED",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=ES_conc_buffered.index,
                y=abs(ES_conc[key].astype(float)),
                mode="lines",
                name=f"ES {key}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=RZ_conc_buffered.index,
                y=abs(RZ_conc[key].astype(float)),
                mode="lines",
                name=f"RZ {key}",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=ES_conc_buffered.index,
            y=abs(ES_conc["H2O_sew"].astype(float) * 1_000_000),
            mode="lines",
            name=f"ES H2O_sew",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=RZ_conc_buffered.index,
            y=abs(RZ_conc["H2O_sew"].astype(float) * 1_000_000),
            mode="lines",
            name=f"RZ H2O_sew",
        )
    )

    pio.show(fig, renderer="browser")


def compare_concentrations_SIMULATION():
    ES_conc_buffered = pd.read_csv(
        rf"output_effluent\ES_buffered_concentrations.csv",
        index_col=0,
        parse_dates=True,
    )
    RZ_conc_buffered = pd.read_csv(
        rf"output_effluent\RZ_buffered_concentrations.csv",
        index_col=0,
        parse_dates=True,
    )
    ES_conc_buffered_RTC = pd.read_csv(
        rf"output_effluent\ES_RTC_buffer_concentrations.csv",
        index_col=0,
    )
    RZ_conc_buffered_RTC = pd.read_csv(
        rf"output_effluent\RZ_RTC_buffer_concentrations.csv",
        index_col=0,
    )
    ES_conc = pd.read_csv(rf"output_effluent\ES.Effluent.csv", index_col=0)
    RZ_conc = pd.read_csv(rf"output_effluent\RZ.Effluent.csv", index_col=0)

    df_west = pd.read_csv(
        rf"data\WEST\SWMM_inputs_dwf_and_precipitation\concentration_check.out.txt",
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
                y=abs(df_west[key].astype(float)),
                mode="lines",
                name=f"West {key}",
            )
        )

    for key in ["COD_part", "COD_sol", "X_TSS_sew", "NH4_sew", "PO4_sew"]:
        fig.add_trace(
            go.Scatter(
                x=ES_conc_buffered.index,
                y=abs(ES_conc_buffered[key].astype(float)),
                mode="lines",
                name=f"ES {key} BUFFERED",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=RZ_conc_buffered.index,
                y=abs(RZ_conc_buffered[key].astype(float)),
                mode="lines",
                name=f"RZ {key} BUFFERED",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=ES_conc_buffered.index,
                y=abs(ES_conc[key].astype(float)),
                mode="lines",
                name=f"ES {key}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=RZ_conc_buffered.index,
                y=abs(RZ_conc[key].astype(float)),
                mode="lines",
                name=f"RZ {key}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ES_conc_buffered.index,
                y=abs(ES_conc_buffered_RTC[key].astype(float)),
                mode="lines",
                name=f"ES RTC {key}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ES_conc_buffered.index,
                y=abs(RZ_conc_buffered_RTC[key].astype(float)),
                mode="lines",
                name=f"RZ RTC {key}",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=ES_conc_buffered.index,
            y=abs(ES_conc["H2O_sew"].astype(float)),
            mode="lines",
            name=f"ES H2O_sew",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ES_conc_buffered.index,
            y=abs(RZ_conc["H2O_sew"].astype(float)),
            mode="lines",
            name=f"RZ H2O_sew",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ES_conc_buffered.index,
            y=abs(ES_conc_buffered["H2O_sew"].astype(float) * 1_000_000),
            mode="lines",
            name=f"ES BUFFERED H2O_sew",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ES_conc_buffered.index,
            y=abs(ES_conc_buffered["H2O_sew"].astype(float) * 1_000_000),
            mode="lines",
            name=f"RZ BUFFERED H2O_sew",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ES_conc_buffered.index,
            y=abs(ES_conc_buffered_RTC["H2O_sew"].astype(float) * 1_000_000),
            mode="lines",
            name=f"ES RTC BUFFERED H2O_sew",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ES_conc_buffered.index,
            y=abs(ES_conc_buffered_RTC["H2O_sew"].astype(float) * 1_000_000),
            mode="lines",
            name=f"RZ RTC BUFFERED H2O_sew",
        )
    )

    pio.show(fig, renderer="browser")


def analyse_concentrate_out():
    """
    This will show no meaningfull difference in outflow concentration.
    Why? Because the base (no rtc) swmm file mimics the behaviour of the WEST model,
    where the storage in ES, never goes below 6600 m3 -> so in SWMM the same behaviour. In RZ(1400 m3).
    Therefor there constant diluation and averaging of the outflows
    """
    # RTC is regular RTC
    # Base is dwf + precipitation (with pollutant mixing)
    # base 2 is dwf + precipitation no mixing
    # dry is only dwf
    # constant is only constant flow

    RTC = pd.read_csv(
        rf"output_swmm\05-03_11-42_out_ES_RTC.csv",
        index_col=0,
        delimiter=";",
        decimal=",",
    )
    start_date = pd.Timestamp("2024-01-01")
    RTC["timestamp"] = start_date + pd.to_timedelta(RTC.index.astype(float), unit="D")
    RTC.set_index("timestamp", inplace=True)
    base = pd.read_csv(
        rf"output_swmm\05-03_11-54_out_ES_no_RTC.csv",
        index_col=0,
        delimiter=";",
        decimal=",",
    )
    start_date = pd.Timestamp("2024-01-01")
    base["timestamp"] = start_date + pd.to_timedelta(base.index.astype(float), unit="D")
    base.set_index("timestamp", inplace=True)

    base2 = pd.read_csv(
        rf"output_swmm\05-03_18-25_out_ES_no_RTC.csv",
        index_col=0,
        delimiter=";",
        decimal=",",
    )
    start_date = pd.Timestamp("2024-01-01")
    base2["timestamp"] = start_date + pd.to_timedelta(
        base2.index.astype(float), unit="D"
    )
    base2.set_index("timestamp", inplace=True)

    dry = pd.read_csv(
        rf"output_swmm\05-03_17-54_out_ES_no_RTC_no_rain.csv",
        index_col=0,
        delimiter=";",
        decimal=",",
    )
    start_date = pd.Timestamp("2024-01-01")
    dry["timestamp"] = start_date + pd.to_timedelta(dry.index.astype(float), unit="D")
    dry.set_index("timestamp", inplace=True)

    constant = pd.read_csv(
        rf"output_swmm\05-03_18-13_out_ES_no_RTC_no_rain_constant.csv",
        index_col=0,
        delimiter=";",
        decimal=",",
    )
    start_date = pd.Timestamp("2024-01-01")
    constant["timestamp"] = start_date + pd.to_timedelta(
        constant.index.astype(float), unit="D"
    )
    constant.set_index("timestamp", inplace=True)

    fig = go.Figure()
    for key in RTC.keys():
        fig.add_trace(
            go.Scatter(
                x=RTC.index,
                y=abs(RTC[key].astype(float)),
                mode="lines",
                name=f"RTC {key}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dry.index,
                y=abs(dry[key].astype(float)),
                mode="lines",
                name=f"dry {key}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=base.index,
                y=abs(base[key].astype(float)),
                mode="lines",
                name=f"base {key}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=base2.index,
                y=abs(base2[key].astype(float)),
                mode="lines",
                name=f"base fixed {key}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=constant.index,
                y=abs(constant[key].astype(float)),
                mode="lines",
                name=f"constant {key}",
            )
        )
        if not "H2O" in key:
            fig.add_trace(
                go.Scatter(
                    x=RTC.index,
                    y=abs(RTC[key].astype(float) / RTC["H2O_sew"].astype(float)),
                    mode="lines",
                    name=f"CONC. RTC {key}",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=base.index,
                    y=abs(base[key].astype(float) / base["H2O_sew"].astype(float)),
                    mode="lines",
                    name=f"CONC. base {key}",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=base2.index,
                    y=abs(base2[key].astype(float) / base2["H2O_sew"].astype(float)),
                    mode="lines",
                    name=f"CONC. base fixed {key}",
                )
            )
    pio.show(fig, renderer="browser")


def compare_forecast_analysis():
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    ensembles = pd.read_csv(
        rf"rain_prediction_log_ensemble.csv", index_col=0, parse_dates=True
    )
    perfect = pd.read_csv(
        rf"rain_prediction_log_perfect.csv", index_col=0, parse_dates=True
    )
    ensembles.columns = [f"{col}_ensemble" for col in ensembles.columns]
    perfect.columns = [f"{col}_perfect" for col in perfect.columns]
    combined = pd.concat([ensembles, perfect], axis=1)
    combined = combined.dropna()

    # combined = combined.loc['2024-09-01': '2024-10-01']

    combined = combined.sort_index()

    # Boolean series: True if predictions match, False otherwise
    es_match = combined["ES_predicted_ensemble"] == combined["ES_predicted_perfect"]
    rz_match = combined["RZ_predicted_ensemble"] == combined["RZ_predicted_perfect"]

    # Add to DataFrame for inspection (optional)
    combined["ES_match"] = es_match
    combined["RZ_match"] = rz_match
    es_accuracy = es_match.mean()  # mean of boolean series gives proportion of True
    rz_accuracy = rz_match.mean()
    print(f"ES forecast agreement: {es_accuracy:.2%}")
    print(f"RZ forecast agreement: {rz_accuracy:.2%}")
    # Drop rows where any of the relevant columns have NaN
    es_valid = combined[["ES_predicted_ensemble", "ES_predicted_perfect"]].dropna()
    rz_valid = combined[["RZ_predicted_ensemble", "RZ_predicted_perfect"]].dropna()

    # Extract clean true/pred arrays
    es_true = es_valid["ES_predicted_perfect"]
    es_pred = es_valid["ES_predicted_ensemble"]

    rz_true = rz_valid["RZ_predicted_perfect"]
    rz_pred = rz_valid["RZ_predicted_ensemble"]

    # Now compute confusion matrices safely
    from sklearn.metrics import confusion_matrix

    es_cm = confusion_matrix(es_true, es_pred, labels=[1, 0])
    rz_cm = confusion_matrix(rz_true, rz_pred, labels=[1, 0])

    print("Confusion Matrix for ES:\n", es_cm)
    print("Confusion Matrix for RZ:\n", rz_cm)

    import pandas as pd
    import numpy as np
    from scipy.ndimage import label

    def get_event_labels(series):
        """Label contiguous 1s in a boolean series."""
        return label(series.values)[0]

    def get_event_ranges(labels, index):
        """Return dict of event_id -> (start_time, end_time)"""
        ranges = {}
        for eid in np.unique(labels):
            if eid == 0:
                continue
            positions = np.where(labels == eid)[0]
            start = index[positions[0]]
            end = index[positions[-1]]
            ranges[eid] = (start, end)
        return ranges

    def match_events_with_tn(pred_series, truth_series):
        index = pred_series.index
        pred_labels = get_event_labels(pred_series == 1)
        truth_labels = get_event_labels(truth_series == 1)

        pred_ranges = get_event_ranges(pred_labels, index)
        truth_ranges = get_event_ranges(truth_labels, index)

        pred_event_ids = set(pred_ranges.keys())
        truth_event_ids = set(truth_ranges.keys())

        true_positive_preds = set()
        true_positive_truths = set()

        for pred_id in pred_event_ids:
            pred_mask = pred_labels == pred_id
            overlapping_truth_ids = set(np.unique(truth_labels[pred_mask])) - {0}
            if overlapping_truth_ids:
                true_positive_preds.add(pred_id)
                true_positive_truths.update(overlapping_truth_ids)

        false_positive_preds = pred_event_ids - true_positive_preds
        false_negative_truths = truth_event_ids - true_positive_truths

        # True Negative Detection — label 0 blocks as events
        pred_inv_labels = get_event_labels(pred_series == 0)
        truth_inv_labels = get_event_labels(truth_series == 0)

        tn_ranges = []
        pred_inv_ids = np.unique(pred_inv_labels)
        for inv_id in pred_inv_ids:
            if inv_id == 0:
                continue
            pred_mask = pred_inv_labels == inv_id
            overlap_truth_ids = set(np.unique(truth_inv_labels[pred_mask])) - {0}
            if overlap_truth_ids:
                # Find start/end of TN range
                positions = np.where(pred_mask)[0]
                start = index[positions[0]]
                end = index[positions[-1]]
                tn_ranges.append((start, end))

        result = {
            "TP_events": len(true_positive_preds),
            "FP_events": len(false_positive_preds),
            "FN_events": len(false_negative_truths),
            "TN_events": len(tn_ranges),
            "Precision": len(true_positive_preds)
            / (len(true_positive_preds) + len(false_positive_preds) + 1e-9),
            "Recall": len(true_positive_preds)
            / (len(true_positive_preds) + len(false_negative_truths) + 1e-9),
            "TP_ranges": [pred_ranges[i] for i in sorted(true_positive_preds)],
            "FP_ranges": [pred_ranges[i] for i in sorted(false_positive_preds)],
            "FN_ranges": [truth_ranges[i] for i in sorted(false_negative_truths)],
            "TN_ranges": tn_ranges,
        }

        return result

    # Example usage:
    print("Eindhoven")
    es_eval = match_events_with_tn(
        combined["ES_predicted_ensemble"], combined["ES_predicted_perfect"]
    )
    print("TP:", es_eval["TP_events"])
    print("FP:", es_eval["FP_events"])
    print("FN:", es_eval["FN_events"])
    print("TN:", es_eval["TN_events"])
    print("RZ")
    rz_eval = match_events_with_tn(
        combined["RZ_predicted_ensemble"], combined["RZ_predicted_perfect"]
    )
    print("TP:", rz_eval["TP_events"])
    print("FP:", rz_eval["FP_events"])
    print("FN:", rz_eval["FN_events"])
    print("TN:", rz_eval["TN_events"])

    fig = go.Figure()

    # Add traces for ES
    fig.add_trace(
        go.Scatter(
            x=combined.index,
            y=combined["ES_predicted_ensemble"],
            mode="lines+markers",
            name="ES Ensemble",
            line=dict(color="blue", dash="dot"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=combined.index,
            y=combined["ES_predicted_perfect"],
            mode="lines+markers",
            name="ES Perfect",
            line=dict(color="blue"),
        )
    )

    # Add traces for RZ
    fig.add_trace(
        go.Scatter(
            x=combined.index,
            y=combined["RZ_predicted_ensemble"],
            mode="lines+markers",
            name="RZ Ensemble",
            line=dict(color="green", dash="dot"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=combined.index,
            y=combined["RZ_predicted_perfect"],
            mode="lines+markers",
            name="RZ Perfect",
            line=dict(color="green"),
        )
    )

    # Update layout
    fig.update_layout(
        title="Rain Forecast Comparison: Ensemble vs. Perfect",
        xaxis_title="Time",
        yaxis_title="Rain Forecast (0 = No, 1 = Yes)",
        yaxis=dict(tickvals=[0, 1]),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    pio.show(fig, renderer="browser")


def river_water_ph_temp():
    df_west = pd.read_csv(
        f"data\WEST\WEST_modelRepository\Model_Dommel_Full\comparison.out.txt",
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
    
    df_west = df_west.loc['2024-04-15':'2024-10-15']
    
    df_west[".S031.T_wat"].astype(float).plot()
    df_west['.S031.pH'].plot()
    df_west[".S031.T_wat"].astype(float).mean()