import swmm_api as sa
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


output_rtc = sa.SwmmOutput(rf"data\SWMM\model_jip.out").to_frame()
output = sa.SwmmOutput(rf"data\SWMM\model_jip_no_rtc.out").to_frame()

output_rtc_jul = output_rtc.loc["2024-07-01":"2024-08-01"]
output_jul = output.loc["2024-07-01":"2024-08-01"]

nodes = ["P_eindhoven_out", "P_riool_zuid_out"]
title = ["Pump outflow Eindhoven", "Pump outflow Riool Zuid"]

for i, node in enumerate(nodes):
    plt.figure(figsize=(20, 4))
    output_rtc_jul.link[node].flow.plot(label="RTC flow")
    output_jul.link[node].flow.plot(label="Regular flow")
    plt.grid()
    plt.xlabel("Date")
    plt.ylabel("Outflow [cms]")
    plt.title(title[i])
    plt.legend()


# OK ramping, but bad return ramp
output_rtc_event = output_rtc.loc["2024-02-05":"2024-02-11"]
output_event = output.loc["2024-02-05":"2024-02-11"]
nodes = ["P_eindhoven_out", "P_riool_zuid_out"]
title = ["Pump outflow Eindhoven", "Pump outflow Riool Zuid"]

for i, node in enumerate(nodes):
    plt.figure(figsize=(10, 7))
    output_rtc_event.link[node].flow.plot(label="RTC flow")
    output_rtc_event.node["pipe_ES"].depth.plot(label="RTC depth")
    output_event.link[node].flow.plot(label="Regular flow")
    output_event.node["pipe_ES"].depth.plot(label="Regular depth")
    plt.grid()
    plt.xlabel("Date")
    plt.ylabel("Outflow [cms]")
    plt.title(title[i])
    plt.legend()


# not great ramping
output_rtc_event = output_rtc.loc["2024-08-19":"2024-08-23"]
output_event = output.loc["2024-08-19":"2024-08-23"]
nodes = ["P_eindhoven_out", "P_riool_zuid_out"]
title = ["Pump outflow Eindhoven", "Pump outflow Riool Zuid"]

for i, node in enumerate(nodes):
    plt.figure(figsize=(10, 7))
    output_rtc_event.link[node].flow.plot(label="RTC flow")
    output_rtc_event.node["pipe_ES"].depth.plot(label="RTC depth")
    output_event.link[node].flow.plot(label="Regular flow")
    output_event.node["pipe_ES"].depth.plot(label="Regular depth")
    plt.grid()
    plt.xlabel("Date")
    plt.ylabel("Outflow [cms]")
    plt.title(title[i])
    plt.legend()
