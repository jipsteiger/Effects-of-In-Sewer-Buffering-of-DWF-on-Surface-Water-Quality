import swmm_api as sa
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio
from plotly.subplots import make_subplots


class PostProcess:
    def __init__(self, model_name):
        self.model_name = model_name
        self.report = sa.read_rpt_file(rf"data\SWMM\{model_name}.rpt")
        self.output = sa.SwmmOutput(rf"data\SWMM\{model_name}.out").to_frame()
        self.current_time = datetime.now().strftime("%m-%d_%H-%M")

    def create_outfall_csv(self):
        for outfall in ["out_RZ", "out_ES"]:
            outfall_timeseries = self.output["node"][outfall]["total_inflow"]
            outfall_timeseries.to_csv(f"swmm_output/{self.current_time}_{outfall}.csv")

    def plot_outfalls(self, save=False, plot_rain=False):
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
        )

    def plot_pumps(self, save=False, plot_rain=False):
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
                filename=f"swmm_output/plots/{self.current_time}_{file_name}.html",
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
