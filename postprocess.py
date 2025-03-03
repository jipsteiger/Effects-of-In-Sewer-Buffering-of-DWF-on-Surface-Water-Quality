import swmm_api as sa
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


class PostProcess:
    def __init__(self, model_name):
        self.model_name = model_name
        self.report = sa.read_rpt_file(rf"data\SWMM\{model_name}.rpt")
        self.output = sa.SwmmOutput(rf"data\SWMM\{model_name}.out").to_frame()
        self.current_time = datetime.now().strftime("%m-%d_%H-%M")

    def create_outfall_txt(self, suffix=""):
        for outfall in ["out_RZ", "out_ES"]:
            outfall_timeseries = pd.DataFrame(
                self.output["node"][outfall]["total_inflow"]
            )
            start_date = outfall_timeseries.index[0]
            outfall_timeseries[".t"] = (
                outfall_timeseries.index - start_date
            ).total_seconds() / (24 * 60 * 60)
            outfall_timeseries[f".in_{outfall}"] = outfall_timeseries["total_inflow"]
            output_file = f"#.t\t.in_{outfall}\n#d\tCMS\n"
            output_file += outfall_timeseries[[".t", f".in_{outfall}"]].to_csv(
                index=False, sep="\t", header=False
            )
            with open(
                f"swmm_output/{self.current_time}_{outfall}_{suffix}.txt", "w"
            ) as f:
                f.write(output_file)
            outfall_timeseries["total_inflow"].to_csv(
                f"swmm_output/{self.current_time}_{outfall}_{suffix}.csv",
                sep=";",
                decimal=",",
            )

    def plot_outfalls(self, save=False, plot_rain=False, suffix=""):
        outfalls = set(
            [
                node[0]
                for node in list(self.output.node.keys())
                if ("cso" in node[0]) or ("out" in node[0])
            ]
        )
        self.plot(
            outfalls,
            "node",
            "total_inflow",
            "outfalls",
            "WWTP Inlet flows",
            "Datetime",
            "Flow [CMS]",
            save=save,
            plot_rain=plot_rain,
            suffix=suffix,
        )

    def plot_pumps(self, save=False, plot_rain=False, suffix=""):
        pumps = [
            "P_riool_zuid_out",
            "P_eindhoven_out",
            "P_aalst_1",
            "P_aalst_2",
            "P_aalst_3",
            "P_aalst_4",
        ]
        self.plot(
            pumps,
            "link",
            "flow",
            "pumps",
            "Pump flows",
            "Datetime",
            "Flow [CMS]",
            save=save,
            plot_rain=plot_rain,
            suffix=suffix,
        )

    def plot(
        self,
        ids,
        object_type,
        variable,
        file_name,
        title,
        xaxis,
        yaxis,
        save,
        plot_rain,
        suffix,
    ):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        layout_config = dict(
            title=title,
            xaxis_title=xaxis,
            yaxis_title=yaxis,
            hovermode="x unified",
        )
        for id in ids:
            fig = self.add_trace(fig, object_type, id, variable)

        if plot_rain:
            fig = self.add_rain(fig)
            layout_config["yaxis2"] = dict(
                title="Rainfall (mm)",
                side="right",
            )

        fig.update_layout(**layout_config)
        if save:
            pyo.plot(
                fig,
                filename=f"swmm_output/plots/{self.current_time}_{file_name}_{suffix}.html",
                auto_open=True,
            )
        else:
            pio.show(fig, renderer="browser")

    def add_trace(
        self,
        fig,
        object_type,
        id,
        variable,
        secondary=False,
        name=None,
        mode="lines",
        marker=None,
    ):
        """Adds a trace to the plot"""
        if not name:
            name = id
        fig.add_trace(
            go.Scatter(
                x=self.output[object_type][id][variable].index,
                y=self.output[object_type][id][variable],
                mode=mode,
                name=name,
                marker=marker,
            ),
            secondary_y=secondary,
        )
        return fig

    def add_rain(self, fig):
        """Adds rainfall data to the second y-axis"""
        locations = {"ES": 16, "GE": 1285, "RZ1": 144, "RZ2": 120}
        for location, catchment in locations.items():
            fig = self.add_trace(
                fig,
                "subcatchment",
                f"c_{catchment}_catchment",
                "rainfall",
                secondary=True,
                name=f"rain_{location}",
                mode="markers",
                marker=dict(symbol="x"),
            )
            fig.data[-1].visible = "legendonly"
        return fig


def plot_make_dwf_only():
    current_time = datetime.now().strftime("%m-%d_%H-%M")
    aa = sa.SwmmOutput(rf"data\SWMM\model_jip.out").to_frame()

    for outfall in ["out_RZ", "out_ES"]:
        aa.node[outfall]["total_inflow"].plot()
        timeseries = aa.node[outfall]["total_inflow"]
        value = timeseries.sum() / len(timeseries)
        ideal_dwf = pd.DataFrame([value] * len(timeseries), index=timeseries.index)
        ideal_dwf.plot()

        plt.legend()

        bb = aa.node[f"out_RZ"]["total_inflow"] + aa.node[f"out_ES"]["total_inflow"]
        ideal_inflow = bb.sum() / len(bb)

        ideal_inflow_df = pd.DataFrame([ideal_inflow] * len(bb), index=bb.index)
        ideal_inflow_df.plot()

        start_date = ideal_inflow_df.index[0]
        ideal_inflow_df[".t"] = (ideal_inflow_df.index - start_date).total_seconds() / (
            24 * 60 * 60
        )
        ideal_inflow_df[f".in_{outfall}"] = ideal_inflow_df.iloc[:, 0]
        output_file = f"#.t\t.in_{outfall}\n#d\tCMS\n"
        output_file += ideal_inflow_df[[".t", f".in_{outfall}"]].to_csv(
            index=False, sep="\t", header=False
        )
        with open(f"swmm_output/{current_time}_{outfall}_ideal_dwf.txt", "w") as f:
            f.write(output_file)
        ideal_inflow_df.iloc[:, 0].to_csv(
            f"swmm_output/{current_time}_{outfall}_ideal_dwf.csv",
            sep=";",
            decimal=",",
        )


import swmm_api as sa
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


def read_WEST_output():
    df_west = pd.read_csv(
        f"data\WEST\Model_Dommel_Full\westcompare.out.txt",
        delimiter="\t",
        header=0,
        index_col=0,
    ).iloc[1:, :]
    start_date = pd.Timestamp("2024-01-01")
    df_west["timestamp"] = start_date + pd.to_timedelta(
        df_west.index.astype(float), unit="D"
    )
    df_west.set_index("timestamp", inplace=True)
    fig = go.Figure()

    # Turn variables to m3/s if applicable
    # Index(['.ES_out.Q_in', '.RZ_out.Q_in', '.ST_106.FillingDegreeIn',
    #    '.ST_106.Q_Out', '.ST_106.Q_out', '.c_106.Rainfall',
    #    '.c_106.comb.Out_1(Rain)', '.c_106.comb2.Q_i', '.c_106.dwf.Q_out',
    #    '.c_106.runoff.In_1(Evaporation)'],
    units = [
        3600,
        3600,
        1,
        3600,
        3600 * 24,
        1,
        24,
        3600 * 24,
        3600 * 24,
        1,
        24,
        10,
        3600 * 24,
        3600 * 24,
    ]
    # 3600,3600,1,3600,3600*24,1,24,3600*24,3600*24,1,24,1,3600*24,3600*24

    for key, unit in zip(df_west.keys(), units):
        if key == ".c_106.runoff.In_1(Evaporation)":
            continue
        fig.add_trace(
            go.Scatter(
                x=df_west.index,
                y=df_west[key].astype(float) / unit,
                mode="lines",
                name=f"WEST {key}",
            )
        )

    output = sa.SwmmOutput(rf"data\SWMM\model_jip_WEST_data_replicated.out").to_frame()
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

    pio.show(fig, renderer="browser")

    pyo.plot(
        fig,
        filename=f"testing.html",
        auto_open=True,
    )


# print('Small rain event during a single night')
# print(output.node["c_106_vr_storage_13"].loc['2024-03-04':'2024-03-06',"total_inflow"].sum())
# print((df_west.loc['2024-03-04':'2024-03-06','.c_106.comb2.Q_i'].astype(float) / 3600 /24).sum() *3)
# print((df_west.loc['2024-03-04':'2024-03-06','.ST_106.Q_Out'].astype(float) / 3600 ).sum() *3)

# print('Dry day:')
# print(output.node["c_106_vr_storage_13"].loc['2024-06-26',"total_inflow"].sum())
# print((df_west.loc['2024-06-26','.ST_106.Q_Out'].astype(float) / 3600).sum() *3)

# print('Medium event over multiple days')
# print(output.node["c_106_vr_storage_13"].loc['2024-04-08':'2024-04-14',"total_inflow"].sum())
# print((df_west.loc['2024-04-08':'2024-04-14','.c_106.comb2.Q_i'].astype(float) / 3600 /24).sum() *3)
# print((df_west.loc['2024-04-08':'2024-04-14','.ST_106.Q_Out'].astype(float) / 3600 ).sum() *3)
