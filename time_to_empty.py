import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

P_ES_MAX = 5.5556  # CMS
P_RZ_MAX = 3.3000  # CMS

V_ES_MAX = 250_000  # CM
V_RZ_MAX = 167_000  # Cm


def blae(pump_rate, volume, timeseries, name):
    timeseries[f"{name}_{pump_rate:.3f}"] = []
    while volume > 0:
        timeseries[f"{name}_{pump_rate:.3f}"].append(volume)
        volume -= pump_rate * 3600
        print(volume)

    return timeseries


def ooo(P, V, name):
    timeseries = {}
    if name == "ES":
        P_min = 1.3
    elif name == "RZ":
        P_min = 0.7
    while P > P_min:
        timeseries = blae(P, V, timeseries, name)
        P = P - 0.3
    return timeseries


timeseries_ES = ooo(P_ES_MAX, V_ES_MAX, "ES")
timeseries_RZ = ooo(P_RZ_MAX, V_RZ_MAX, "RZ")

df_ES = pd.DataFrame.from_dict(timeseries_ES, orient="index").transpose()
df_RZ = pd.DataFrame.from_dict(timeseries_RZ, orient="index").transpose()
