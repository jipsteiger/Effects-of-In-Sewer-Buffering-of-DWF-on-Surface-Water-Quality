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


def plot_all():
    model_names = [
        "Pollutant_no_RTC",
        "Pollutant_No_RTC_no_rain",
        "Pollutant_No_RTC_no_rain_constant",
        "Pollutant_RTC",
        "Pollutant_RTC_ensemble",
        "WEST_modelRepository/Model_Dommel_Full",
    ]
    short_names = {
        "Pollutant_no_RTC": "No RTC",
        "Pollutant_No_RTC_no_rain": "No RTC, No Rain",
        "Pollutant_No_RTC_no_rain_constant": "No RTC, Const. Load",
        "Pollutant_RTC": "RTC",
        "Pollutant_RTC_ensemble": "RTC Ensemble",
        "WEST_modelRepository/Model_Dommel_Full": "Reference",
    }

    model_data = {}
    fig = go.Figure()
    fig_conc = go.Figure()

    for model_name in model_names:
        data = pd.read_csv(
            rf"data\WEST\{model_name}\comparison.out.txt",
            delimiter="\t",
            header=0,
            index_col=0,
            low_memory=False,
        ).iloc[1:, :]
        start_date = pd.Timestamp("2024-01-01")
        data["timestamp"] = start_date + pd.to_timedelta(
            data.index.astype(float), unit="D"
        )
        data.set_index("timestamp", inplace=True)
        data = data.loc["2024-04-15":"2024-10-16"]
        model_data[model_name] = data

        for key in data.keys():
            if "WWTP2river" in key:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=abs(data[key].astype(float)),
                        mode="lines",
                        name=f"{short_names[model_name]} {key}",
                    )
                )
                if "H2O" in key:
                    denominator = 1
                else:
                    denominator = data[".WWTP2river.Outflow(rH2O)"].astype(float)
                fig_conc.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=abs(data[key].astype(float) / denominator),
                        mode="lines",
                        name=f"{short_names[model_name]} Conc. {key}",
                    )
                )
    pio.show(fig, renderer="browser")
    pio.show(fig_conc, renderer="browser")
