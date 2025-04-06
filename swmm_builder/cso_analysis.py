import swmm_api as sa
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio


class CSO_analysis:
    def __init__(self):
        self.output = sa.SwmmOutput(
            rf"data\SWMM\model_jip_WEST_regen_geen_extra_storage_ts5.out"  # COPY IS SELECTED
        ).to_frame()
        self.csos_swmm = [
            "cso_ES_1",
            "cso_RZ",
            [
                "cso_AALST",
                "cso_c_123",
                "cso_c_99",
                "cso_c_112",
                "cso_c_119_1",
                "cso_c_119_2",
                "cso_c_119_3",
                "cso_c_122",
            ],
            "cso_gb_136",
            "cso_Geldrop",
        ]

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
        df_west[".Waalre.Q_in"] = df_west[".Waalre.Q_in"].astype(float) / 24
        df_west.set_index("timestamp", inplace=True)
        self.df_west = df_west.loc[self.output.index[0] : self.output.index[-1], :]

        self.west_cso_keys()

    def west_cso_keys(self):
        Q_keys = self.df_west.keys()

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
        ]

        self.csos = [
            cso_ES_1_keys,
            cso_RZ_keys,
            cso_AALST_keys,
            cso_gb_136_keys,
            cso_Geldrop_keys,
        ]

    def ES_analysis(self):
        cso_west = self.csos[0]
        output = self.output
        df_west = self.df_west
        cso_swmm = self.csos_swmm[0]
        cso_name = "ES_1"

        cso_swmm_flow = output.node[cso_swmm].total_inflow * 60 * 60  # cms to cmh
        cso_west_flow = df_west[cso_west].astype(float).sum(axis=1)
        fig = go.Figure()

        other_values = df_west[
            [
                ".c_24_sp.Q_i",
                ".pipe_ES.FillingDegreeTank",
                ".pipe_ES.Q_i",
                ".pipe_ES.V",
            ]
        ]
        fig.add_trace(
            go.Scatter(
                x=output.subcatchment["c_106_catchment"]["rainfall"].index,
                y=output.node["pipe_ES"].volume / 165000,
                mode="lines",
                name=f"SWMM ES Filling degree",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=output.subcatchment["c_106_catchment"]["rainfall"].index,
                y=output.node["pipe_ES"].volume,
                mode="lines",
                name=f"SWMM ES V",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=output.subcatchment["c_106_catchment"]["rainfall"].index,
                y=output.node["pipe_ES"].total_inflow * 3600,
                mode="lines",
                name=f"SWMM ES Inflow",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=output.subcatchment["c_106_catchment"]["rainfall"].index,
                y=output.link["P_eindhoven_out"].flow * 3600,
                mode="lines",
                name=f"SWMM ES Pump flow",
            )
        )

        for key in other_values:
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
                x=cso_swmm_flow.index,
                y=cso_swmm_flow.values.astype(float),
                mode="lines",
                name=f"SWMM {cso_swmm}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cso_west_flow.index,
                y=cso_west_flow.values.astype(float),
                mode="lines",
                name=f"WEST {cso_name}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_west.index,
                y=df_west[".c_24.Outflow(H2O_sew)"].values.astype(float)
                / 24
                / 1e6
                * -1,
                mode="lines",
                name=f"WEST .c_24 Outflow",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=output.subcatchment["c_106_catchment"]["rainfall"].index,
                y=output.node["c_24_vr_storage_5"].lateral_inflow * 3600,
                mode="lines",
                name=f"SWMM ES c_24 lat inflow",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_west.index,
                y=df_west[".pipe_ES.Q_over"].values.astype(float),
                mode="lines",
                name=f"WEST .pipe_ES.Q_over",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_west.index,
                y=df_west[".pipe_ES.Q_pump_help"].values.astype(float) / 24,
                mode="lines",
                name=f"WEST .pipe_ES.Q_pump_help",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_west.index,
                y=df_west[".pipe_ES_tr.Q_i"].values.astype(float) / 24,
                mode="lines",
                name=f"WEST .pipe_ES_tr.Q_i",
            )
        )
        fig.update_layout(title_text=f"Eindhoven stad")

        pio.show(fig, renderer="browser")

        print("Outflow to WWTP")
        swmm = (
            output.link["P_eindhoven_out"].flow["2024-05-20":"2024-05-22"].sum()
            / 12
            * 3600
        )
        west = (
            df_west.loc["2024-05-20":"2024-05-22", ".pipe_ES_tr.Q_i"]
            .astype(float)
            .sum()
            / 6
            / 24
        )
        print(f"{swmm=:.2f}")
        print(f"{west=:.2f}")
        print("\nInflow to storage")
        swmm = (
            output.node["pipe_ES"].total_inflow["2024-05-20":"2024-05-22"].sum()
            / 12
            * 3600
        )
        west = (
            df_west.loc["2024-05-20":"2024-05-22", ".pipe_ES.Q_i"].astype(float).sum()
            / 6
        )
        print(f"{swmm=:.2f}")
        print(f"{west=:.2f}")
        print("\nCSO outflow")
        swmm = (
            output.node["cso_ES_1"].total_inflow["2024-05-20":"2024-05-22"].sum()
            / 12
            * 3600
        )
        west = (
            df_west.loc["2024-05-20":"2024-05-22", ".pipe_ES.Q_over"]
            .astype(float)
            .sum()
            / 6
        )
        print(f"{swmm=:.2f}")
        print(f"{west=:.2f}")

    def RZ_analysis(self):
        cso_swmm_flow = (
            self.output.node[self.csos_swmm[1]].total_inflow * 60 * 60
        )  # cms to cmh
        cso_west_flow = self.df_west[self.csos[1]].astype(float).sum(axis=1)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=cso_west_flow.index,
                y=cso_west_flow.values.astype(float),
                mode="lines",
                name=f"WEST cso flow",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cso_swmm_flow.index,
                y=cso_swmm_flow.values.astype(float),
                mode="lines",
                name=f"SWMM cso flow",
            )
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="",
            legend_title="Transport buis",
            bargap=0.1,  # Reducing gap between bars for better visibility
        )
        pio.show(fig, renderer="browser")

    def GB_analysis(self):

        cso_swmm_flow = (
            self.output.node[self.csos_swmm[3]].total_inflow * 60 * 60
        )  # cms to cmh
        cso_west_flow = self.df_west[self.csos[3]].astype(float).sum(axis=1)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=cso_west_flow.index,
                y=cso_west_flow.values.astype(float),
                mode="lines",
                name=f"WEST cso flow",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cso_swmm_flow.index,
                y=cso_swmm_flow.values.astype(float),
                mode="lines",
                name=f"SWMM cso flow",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=cso_swmm_flow.index,
                y=self.output.node["gb_136"].volume / 8623,
                mode="lines",
                name=f"SWMM GB Filling Degree",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cso_swmm_flow.index,
                y=self.output.node["gb_136"].volume,
                mode="lines",
                name=f"SWMM GB Volume",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cso_swmm_flow.index,
                y=self.output.node["gb_136"].total_inflow * 3600,
                mode="lines",
                name=f"SWMM GB Inflow",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=cso_swmm_flow.index,
                y=self.output.link["Con_1119.1"].flow * 3600,
                mode="lines",
                name=f"SWMM GB Pump flow",
            )
        )
        for value, unit in zip(
            [
                ".GB_136.FillingDegreeTank",
                ".GB_136.Q_Out",
                ".GB_136.Q_i",
                ".GB_136.Q_over",
                ".GB_136.V",
            ],
            [1, 24, 24, 1, 1],
        ):
            fig.add_trace(
                go.Scatter(
                    x=self.df_west.index,
                    y=self.df_west[value].values.astype(float) / unit,
                    mode="lines",
                    name=f"WEST GB {value}",
                )
            )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="",
            legend_title="Mierlo / GB",
            bargap=0.1,  # Reducing gap between bars for better visibility
        )
        pio.show(fig, renderer="browser")

    def GE_analysis(self):
        cso_swmm_flow = (
            self.output.node[self.csos_swmm[4]].total_inflow * 60 * 60
        )  # cms to cmh
        cso_west_flow = self.df_west[self.csos[4]].astype(float).sum(axis=1)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=cso_west_flow.index,
                y=cso_west_flow.values.astype(float),
                mode="lines",
                name=f"WEST cso flow",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cso_swmm_flow.index,
                y=cso_swmm_flow.values.astype(float),
                mode="lines",
                name=f"SWMM cso flow",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=cso_swmm_flow.index,
                y=self.output.node["pipe_Geldrop"].volume / 27800,
                mode="lines",
                name=f"SWMM GE Filling Degree",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cso_swmm_flow.index,
                y=self.output.node["pipe_Geldrop"].volume,
                mode="lines",
                name=f"SWMM GE Volume",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cso_swmm_flow.index,
                y=self.output.node["pipe_Geldrop"].total_inflow * 3600,
                mode="lines",
                name=f"SWMM GE Inflow",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cso_swmm_flow.index,
                y=self.output.link["geldrop_out_north"].flow * 3600,
                mode="lines",
                name=f"SWMM GE Outflow 1",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cso_swmm_flow.index,
                y=self.output.link["geldrop_out_east"].flow * 3600,
                mode="lines",
                name=f"SWMM GE Outflow 2",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=cso_swmm_flow.index,
                y=self.output.node["gb127"].volume / 8600,
                mode="lines",
                name=f"SWMM gb127 Filling Degree",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cso_swmm_flow.index,
                y=self.output.node["gb127"].volume,
                mode="lines",
                name=f"SWMM gb127 Volume",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cso_swmm_flow.index,
                y=self.output.node["gb127"].total_inflow * 3600,
                mode="lines",
                name=f"SWMM gb127 inflow",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=cso_swmm_flow.index,
                y=self.output.node["cso_gb127"].total_inflow * 3600,
                mode="lines",
                name=f"SWMM gb127 cso",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=cso_swmm_flow.index,
                y=self.output.node["cso_Geldrop"].total_inflow * 3600,
                mode="lines",
                name=f"SWMM Geldrop cso",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=cso_swmm_flow.index,
                y=(
                    self.output.node["cso_Geldrop"].total_inflow
                    + self.output.node["cso_gb127"].total_inflow
                )
                * 3600,
                mode="lines",
                name=f"SWMM Combined geldrop CSO",
            )
        )

        for value, unit in zip(
            [
                ".GB_128.FillingDegreeTank",
                ".GB_128.Q_Out",
                ".GB_128.Q_i",
                ".GB_128.Q_over",
                ".GB_128.V",
            ],
            [1, 24, 1, 24, 1],
        ):
            fig.add_trace(
                go.Scatter(
                    x=self.df_west.index,
                    y=self.df_west[value].values.astype(float) / unit,
                    mode="lines",
                    name=f"WEST GB {value}",
                )
            )
        for value, unit in zip(
            [
                ".GB_127.FillingDegreeTank",
                ".GB_127.Q_Out",
                ".GB_127.Q_i",
                ".GB_127.Q_over",
                ".GB_127.V",
            ],
            [1, 1, 1, 24, 1],
        ):
            fig.add_trace(
                go.Scatter(
                    x=self.df_west.index,
                    y=self.df_west[value].values.astype(float) / unit,
                    mode="lines",
                    name=f"WEST GB {value}",
                )
            )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="",
            legend_title="Geldrop",
            bargap=0.1,  # Reducing gap between bars for better visibility
        )
        pio.show(fig, renderer="browser")

    def aalst_analysis(self):
        cso_west_flow = (
            self.df_west[self.csos[2] + [".Waalre.Q_in", ".Aalst.Q_in"]]
            .astype(float)
            .sum(axis=1)
        )

        fig = go.Figure()

        for cso in self.csos_swmm[2]:
            fig.add_trace(
                go.Scatter(
                    x=self.output.index,
                    y=self.output.node[cso].total_inflow.values.astype(float) * 3600,
                    mode="lines",
                    name=f"SWMM cso flow {cso}",
                )
            )
        combined = (
            self.output.node[self.csos_swmm[2][0]].total_inflow.values
            + self.output.node[self.csos_swmm[2][1]].total_inflow.values
            + self.output.node[self.csos_swmm[2][2]].total_inflow.values
            + self.output.node[self.csos_swmm[2][3]].total_inflow.values
            + self.output.node[self.csos_swmm[2][4]].total_inflow.values
            + self.output.node[self.csos_swmm[2][5]].total_inflow.values
            + self.output.node[self.csos_swmm[2][6]].total_inflow.values
        ) * 3600

        fig.add_trace(
            go.Scatter(
                x=self.output.index,
                y=combined,
                mode="lines",
                name=f"SWMM cso flow combined",
            )
        )

        for name, volume in zip(
            ["c_119_1", "c_119_2", "c_119_3"],
            [5000, 7000, 14000],
        ):
            fig.add_trace(
                go.Scatter(
                    x=self.output.index,
                    y=self.output.node[name].volume / volume,
                    mode="lines",
                    name=f"SWMM {name} Filling Degree",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.output.index,
                    y=self.output.node[name].volume,
                    mode="lines",
                    name=f"SWMM {name} Volume",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.output.index,
                    y=self.output.node[name].total_inflow * 3600,
                    mode="lines",
                    name=f"SWMM {name} Inflow",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=cso_west_flow.index,
                y=cso_west_flow.values.astype(float),
                mode="lines",
                name=f"WEST cso flow combined",
            )
        )
        for west_cso in self.csos[2] + [".Waalre.Q_in", ".Aalst.Q_in"]:
            fig.add_trace(
                go.Scatter(
                    x=self.df_west.index,
                    y=self.df_west[west_cso].astype(float),
                    mode="lines",
                    name=f"WEST GB CSO {west_cso}",
                )
            )

        for name in ["BT_119_1", "BT_119_2", "GB_119_3", "Aalst_gemaal"]:
            # for name in ["ST_122"]:
            for value, unit in zip(
                [
                    f".{name}.FillingDegreeTank",
                    f".{name}.Q_Out",
                    f".{name}.Q_i",
                    f".{name}.Q_over",
                    f".{name}.V",
                ],
                [1, 1, 1, 1, 1],
            ):
                fig.add_trace(
                    go.Scatter(
                        x=self.df_west.index,
                        y=self.df_west[value].values.astype(float) / unit,
                        mode="lines",
                        name=f"WEST GB {value}",
                    )
                )

        # fig.add_trace(
        #     go.Scatter(
        #         x=self.df_west.index,
        #         y=self.df_west[[".BT_119_1.V", ".BT_119_2.V", ".GB_119_3.V"]]
        #         .astype(float)
        #         .sum(axis=1)
        #         .values
        #         / unit,
        #         mode="lines",
        #         name=f"WEST GB 119 V",
        #     )
        # )

        # fig.add_trace(
        #     go.Scatter(
        #         x=self.df_west.index,
        #         y=self.df_west[[".BT_119_1.Q_i", ".BT_119_2.Q_i", ".GB_119_3.Q_i"]]
        #         .astype(float)
        #         .sum(axis=1)
        #         .values
        #         / unit,
        #         mode="lines",
        #         name=f"WEST GB 119 Q_i",
        #     )
        # )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="",
            legend_title="Aalst en up",
            bargap=0.1,  # Reducing gap between bars for better visibility
        )

        pio.show(fig, renderer="browser")

    def total_analysis(self):
        west_csos = {
            "ES": self.csos[0],
            "GB": self.csos[3],
            "GE": self.csos[4],
            "TL": self.csos[1],
            "RZ": self.csos[2] + [".Waalre.Q_in", ".Aalst.Q_in"],
        }

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

        fig = go.Figure()
        for key in swmm_csos.keys():
            swmm_values = 0
            for swmm_cso in swmm_csos[key]:
                swmm_values += self.output.node[swmm_cso].total_inflow.values * 3600

            west_values = self.df_west[west_csos[key][:]].astype(float).sum(axis=1)

            fig.add_trace(
                go.Scatter(
                    x=self.output.index,
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


analysis = CSO_analysis()
analysis.ES_analysis()
analysis.RZ_analysis()
analysis.GB_analysis()
analysis.GE_analysis()
analysis.aalst_analysis()
analysis.total_analysis()
