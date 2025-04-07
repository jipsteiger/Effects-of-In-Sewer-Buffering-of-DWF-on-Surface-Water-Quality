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
            dec_index = (
                outfall_timeseries.index - pd.Timestamp("2024-01-01")
            ).total_seconds() / (24 * 60 * 60)
            H2O_sew = (
                outfall_timeseries["total_inflow"] * (24 * 3600)  # to m3/d
            ).values  # From CMS to g/d
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
            df_csv = pd.DataFrame(west_values, index=outfall_timeseries.index)

            output_file = f"#.t\tH2O_sew\tNH4_sew\tPO4_sew\tCOD_sol\tX_TSS_sew\tCOD_part\n#d\tm3/d\tm3/d\tm3/d\tm3/d\tm3/d\tm3/d\n"
            output_file += df[:].to_csv(sep="\t", header=False)
            with open(
                f"swmm_output/{self.current_time}_{outfall}_{suffix}.txt", "w"
            ) as f:
                f.write(output_file)
            df.to_csv(
                f"swmm_output/{self.current_time}_{outfall}_{suffix}.csv",
                sep=";",
                decimal=",",
            )
            static_name = "latest"  # Change this as needed
            with open(f"swmm_output/{static_name}_{outfall}_out.txt", "w") as f:
                f.write(output_file)
            df_csv.to_csv(
                f"swmm_output/{static_name}_{outfall}_out.csv",
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

    def plot_pumps(
        self,
        save=False,
        plot_rain=False,
        suffix="",
        target_setting=False,
        storage=False,
    ):
        pumps = [
            "P_eindhoven_out",
            "P_riool_zuid_out",
            # "P_aalst_1",
            # "P_aalst_2",
            # "P_aalst_3",
            # "P_aalst_4",
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
            target_setting=target_setting,
            storage=storage,
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
        target_setting=False,
        storage=False,
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
        if target_setting:
            fig = self.add_target_settings(fig)
        if storage:
            fig = self.add_storage_depth(fig)

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

    def add_target_settings(self, fig):
        df = pd.read_csv(
            "swmm_output/target_setting.csv", index_col=0, parse_dates=True
        )
        for key in df:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df.loc[:, key].values,
                    mode="lines",
                    name=key,
                    marker=None,
                ),
                secondary_y=False,
            )
        return fig

    def add_storage_depth(self, fig):
        for storage in ["pipe_ES", "pre_ontvangstkelder"]:
            fig.add_trace(
                go.Scatter(
                    x=self.output.index,
                    y=self.output.node[storage].depth.values,
                    mode="lines",
                    name=storage + " depth",
                    marker=None,
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=self.output.index,
                    y=self.output.node[storage].total_inflow.values,
                    mode="lines",
                    name=storage + " total_inflow",
                    marker=None,
                ),
                secondary_y=False,
            )
            fig.data[-1].visible = "legendonly"

        return fig
