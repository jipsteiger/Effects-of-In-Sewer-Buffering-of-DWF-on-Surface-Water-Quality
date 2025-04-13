import swmm_api as sa
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio


class CSO_analysis:
    def __init__(self):
        self.output_before = sa.SwmmOutput(
            rf"data\SWMM\model_jip_no_rtc.out"
        ).to_frame()
        self.output_after = sa.SwmmOutput(rf"data\SWMM\model_jip.out").to_frame()
        self.swmm_csos = {
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

    def total_analysis(self):
        fig = go.Figure()
        for key in self.swmm_csos.keys():
            swmm_values_before = 0
            swmm_values_after = 0
            for swmm_cso in self.swmm_csos[key]:
                swmm_values_before += (
                    self.output_before.node[swmm_cso].total_inflow.values * 3600
                )
                swmm_values_after += (
                    self.output_after.node[swmm_cso].total_inflow.values * 3600
                )

            fig.add_trace(
                go.Scatter(
                    x=self.output_before.index,
                    y=swmm_values_before,
                    mode="lines",
                    name=f"No RTC; {key}",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.output_after.index,
                    y=swmm_values_after,
                    mode="lines",
                    name=f"RTC; {key}",
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
analysis.total_analysis()
