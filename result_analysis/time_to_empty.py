import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

P_ES_MAX = 3.888  # CMS
P_RZ_MAX = 4.7222  # CMS

V_ES_MAX = 165_000  # CM
V_RZ_MAX = 167_000  # Cm


def blae(pump_rate, volume, timeseries, name):
    timeseries[f"{pump_rate:.3f}"] = []
    while volume > 0:
        timeseries[f"{pump_rate:.3f}"].append(volume)
        volume -= pump_rate * 3600

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
#################################
# Pump duration for different V and Pump flows
#################################
# Constants
P_max = 4.725 * 3600  # seconds
V_max = 170_000  # cubic meters

# Pumping rate range (m³/s)
Q_pump = np.linspace(1, 5, 200)

# Time lines to plot (in seconds)
time_hours = np.arange(3, 33, 3)  # hours
time_lines = [t * 3600 for t in time_hours]

# Plotting
plt.figure(figsize=(10, 6))

for t in time_lines:
    volume = Q_pump * t
    plt.plot(Q_pump, volume)

    # Place a label near the end of the line (on the left, since x-axis is flipped)
    x_text = Q_pump[0] - 0.03  # far left, due to inverted x-axis
    y_text = volume[0]

    # Check if the label goes above the upper limit of ylim

    plt.text(x_text, y_text, f"{int(t / 3600)} h", va="center", ha="left", fontsize=9)

# Optional: add max volume as horizontal line
plt.axhline(V_max, color="gray", linestyle="--")

plt.xlabel("Pump Rate (m³/s)")
plt.ylabel("Total Volume Pumped (m³)")
plt.title("Total Volume Pumped vs Pump Rate for Different Durations")
plt.ylim(top=200_000)
plt.grid(True)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()

#####################################
# Total volume stored analysis
#####################################
pattern_RZ = (
    pd.read_csv(
        rf"swmm_output\swmm_base_outflow(BACKUP)\latest_out_RZ_DWF_only_out.csv",
        delimiter=";",
        decimal=",",
        index_col=0,
        parse_dates=True,
    )
    / 24
    / 3600
    * 5
    * 60
)
pattern_ES = (
    pd.read_csv(
        rf"swmm_output\swmm_base_outflow(BACKUP)\latest_out_ES_DWF_only_out.csv",
        delimiter=";",
        decimal=",",
        index_col=0,
        parse_dates=True,
    )
    / 24
    / 3600
    * 5
    * 60
)

ES_inflow = pattern_ES.loc["2023-05-01":"2023-05-07", "H2O_sew"]
ES_mean_inflow = ES_inflow.mean()
RZ_inflow = pattern_RZ.loc["2023-05-01":"2023-05-07", "H2O_sew"]
RZ_mean_inflow = RZ_inflow.mean()

ES_storage = [0]
RZ_storage = [0]
for i in range(1, len(ES_inflow)):
    ES_delta = ES_inflow.iloc[i] - ES_mean_inflow
    ES_current_storage = max(ES_storage[-1] + ES_delta, 0)
    ES_storage.append(ES_current_storage)

    RZ_delta = RZ_inflow.iloc[i] - RZ_mean_inflow
    RZ_current_storage = max(RZ_storage[-1] + RZ_delta, 0)
    RZ_storage.append(RZ_current_storage)

day_hour_formatter = mdates.DateFormatter(
    "%d %H"
)  # Shows "day hour", e.g., "1 12" for May 1st, 12:00

fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(ES_inflow.index, ES_storage, label="ES Storage")
ax.plot(RZ_inflow.index, RZ_storage, label="RZ Storage")
ax.set_xlabel("Day number and hour [dd hh]")
ax.set_ylabel("Buffered storage Volume [m3]")
ax.grid(True)

# Set the major formatter to include both day and hour
ax.xaxis.set_major_formatter(day_hour_formatter)

# Set HourLocator to increase the number of x-ticks
ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))  # Show every hour
# ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # Show every 2 hours (comment/uncomment based on preference)

# Adjust the limits so the x-axis doesn't go beyond the data range
ax.set_xlim([RZ_inflow.index[0], RZ_inflow.index[-1]])

plt.legend()

# Rotate and align x labels
plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.show()

#################################
# Pump duration for different V and Pump flows smaller volume
#################################

# Constants
P_max = 1 * 3600  # seconds
V_max = 15000  # cubic meters

# Pumping rate range (m³/s)
Q_pump = np.linspace(0.01, 0.66, 200)

# Time lines to plot (in seconds)
time_hours = np.arange(2, 13, 2)  # hours
time_lines = [t * 3600 for t in time_hours]

# Plotting
plt.figure(figsize=(10, 6))

for t in time_lines:
    volume = Q_pump * t
    plt.plot(Q_pump, volume, label=f"{int(t / 3600)} h")

    # Place a label near the end of the line (on the left, since x-axis is flipped)
    x_text = Q_pump[0] - 0.03  # far left, due to inverted x-axis
    y_text = volume[0]

    # Check if the label goes above the upper limit of ylim

    # plt.text(x_text, y_text, f"{int(t / 3600)} h", va="center", ha="left", fontsize=9)

# Optional: add max volume as horizontal line
plt.axhline(V_max, color="gray", linestyle="--")

plt.xlabel("Pump Rate on top of DWF inflow (m³/s)")
plt.ylabel("Total Volume Pumped (m³)")
# plt.title("Total Volume Pumped vs Pump Rate for Different Durations")
plt.ylim(top=V_max + 500)
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()
