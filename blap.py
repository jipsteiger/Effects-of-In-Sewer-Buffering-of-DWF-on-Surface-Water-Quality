import pandas as pd
import datetime as dt

import pandas as pd

import matplotlib.pyplot as plt

precipitaiton = pd.read_csv(
    rf"data\precipitation\csv_selected_area_euradclim\2024_5_min_precipitation_data.csv",
    index_col=0,
    parse_dates=True,
)

data = {
    "Region": ["A", "B", "A", "B"],
    "datetime_of_creation": [
        "2024-01-01 00:00:00",
        "2024-01-01 00:00:00",
        "2024-01-01 00:00:00",
        "2024-01-01 00:00:00",
    ],
    "datetime_of_forecast": [
        "2024-01-01 00:00:00",
        "2024-01-01 00:00:00",
        "2024-01-01 01:00:00",
        "2024-01-01 00:00:00",
    ],
    "forecast": [
        [0, 0, 2, 2, 3, 1, 0],
        [5, 1, 2, 2, 0, 1, 0],
        [0, 5, 2, 0, 0, 1, 1],
        [2, 1, 2, 2, 4, 1, 0],
    ],
}

df = pd.DataFrame(data)
df["datetime_of_creation"] = pd.to_datetime(df["datetime_of_creation"])
df["datetime_of_forecast"] = pd.to_datetime(df["datetime_of_forecast"])

# Expand each forecast into 1 row per forecast hour
df_expanded = df.explode("forecast").reset_index(drop=True)

# Add a column indicating the forecast horizon in hours
df_expanded["forecast_hour"] = df_expanded.groupby(
    ["Region", "datetime_of_creation", "datetime_of_forecast"]
).cumcount()

# Calculate the actual forecasted datetime
df_expanded["forecasted_time"] = df_expanded["datetime_of_forecast"] + pd.to_timedelta(
    df_expanded["forecast_hour"], unit="h"
)

# Example current time in the simulation
current_time = pd.Timestamp("2024-01-01 00:00:00")

# Filter: only keep forecasts that were created at the current time
# and that forecast for the next 12 hours
valid_forecasts = df_expanded[
    (df_expanded["datetime_of_creation"] == current_time)
    & (df_expanded["forecasted_time"] <= current_time + pd.Timedelta(hours=12))
]

# -----------------------------


forecasts = pd.read_csv(
    rf"data\precipitation\csv_forecasts\forecast_data.csv", index_col=0
)
forecasts["date"] = pd.to_datetime(forecasts["date"])
forecasts["date_of_forecast"] = pd.to_datetime(forecasts["date_of_forecast"])
forecasts["ensembles"] = forecasts["ensembles"].apply(
    lambda s: [float(x) for x in s.strip("[]").split()]
)


current_time = pd.to_datetime(
    "2024-04-15 18:00:00"
)  # Keep in mind thate only forecasts are made at intervals of 6 hours
upperbound = 48
upperbound_time = current_time + dt.timedelta(hours=upperbound)
filtered_forecast = forecasts[
    (forecasts.date == current_time) & (forecasts.date_of_forecast <= upperbound_time)
]


grouped = filtered_forecast.groupby(["region", "date_of_forecast"])["ensembles"].apply(
    list
)


def should_activate_pump_avg_confidence(
    region,
    forecast_dict,
    current_time,
    hours_ahead=6,
    rain_threshold_per_hour=1.0,
    required_confidence=0.95,
):
    """
    Decide whether to activate a pump based on average rainfall per hour over forecast horizon.
    """
    forecast_times = [current_time + pd.Timedelta(hours=i) for i in range(hours_ahead)]
    print(forecast_times)
    # Collect ensembles across the forecast window
    ensemble_matrix = []

    for t in forecast_times:
        try:
            ensemble_matrix.append(forecast_dict[region][t])
        except KeyError:
            continue
    print(ensemble_matrix)
    # Transpose to get ensemble-wise aggregation: one row per ensemble member
    ensemble_matrix = list(zip(*ensemble_matrix))  # shape: (n_members, hours)
    print(ensemble_matrix)
    # Compute average rain per hour for each ensemble member
    avg_rains = [
        sum(member_forecast) / hours_ahead for member_forecast in ensemble_matrix
    ]
    print(avg_rains)
    # Compute proportion above threshold
    passing_members = sum(avg > rain_threshold_per_hour for avg in avg_rains)
    print(passing_members)
    confidence = passing_members / len(avg_rains)
    print(confidence)

    return confidence >= required_confidence


result = should_activate_pump_avg_confidence("ES", grouped, current_time)


forecast_times = [current_time + pd.Timedelta(hours=i) for i in range(6)]
ensemble_matrix = []

ensemble_matrix = [grouped["ES"][t][0] for t in forecast_times if t in grouped["ES"]]

# Transpose to get per-member forecasts over time
ensemble_matrix = list(zip(*ensemble_matrix))  # shape: (n_members, hours)

# Now compute averages correctly
avg_rains = [sum(member_forecast) / 6 for member_forecast in ensemble_matrix]


time = pd.to_datetime(
    "2024-07-22 18:00:00"
)  # Keep in mind thate only forecasts are made at intervals of 6 hours
upperbound = 48
upperbound_time = time + dt.timedelta(hours=upperbound)
current_forecast = forecasts[
    (forecasts.date == time) & (forecasts.date_of_forecast <= upperbound_time)
]

current_forecast = current_forecast.groupby(["region", "date_of_forecast"])[
    "ensembles"
].apply(lambda x: [item for sublist in x for item in sublist])


precipitation = precipitaiton.resample("h").sum()
precipitation = precipitation.loc["2024-07-22 18:00:00":"2024-07-24 18:00:00", "ES"]

ES = grouped["ES"]
df = pd.DataFrame(ES)
data = df["ensembles"]
labels = df["label"].tolist() if "label" in df.columns else [str(i) for i in df.index]
plt.figure(figsize=(20, 8))
plt.boxplot(data, labels=labels)
plt.plot(precipitation.index, precipitation.values, "r")
plt.xlabel("Group")
plt.xticks(rotation=90)
plt.ylabel("Values")
plt.title("Boxplots of Ensemble Data")
plt.grid(True)
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. Prepare and sort the forecast data
ES = grouped["ES"].reset_index()
ES["timestamp"] = pd.to_datetime(ES["date_of_forecast"])
ES = ES.sort_values("timestamp")

data = ES["ensembles"]
positions = mdates.date2num(
    ES["timestamp"]
)  # Convert datetime to numeric format for plotting

# 2. Create figure and axis
fig, ax = plt.subplots(figsize=(20, 8))

# 3. Plot boxplots with datetime-based positions
ax.boxplot(data, positions=positions, widths=0.1)

# 4. Plot precipitation (make sure index is datetime and aligns with same time range)
ax.plot(precipitation.index, precipitation.values, "r-", label="Precipitation")

# 5. Formatting
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
plt.xticks(rotation=90)
ax.set_xlabel("Time")
ax.set_ylabel("Values (Shared Axis)")
ax.set_title("Ensemble Forecasts and Precipitation (Shared Y-axis)")
ax.grid(True)
ax.legend()
fig.tight_layout()
plt.show()


######################


forecasts = pd.read_csv(
    rf"data\precipitation\csv_forecasts\forecast_data.csv", index_col=0
)
forecasts["date"] = pd.to_datetime(forecasts["date"])
forecasts["date_of_forecast"] = pd.to_datetime(forecasts["date_of_forecast"])
forecasts["ensembles"] = forecasts["ensembles"].apply(
    lambda s: [float(x) for x in s.strip("[]").split()]
)

time = pd.to_datetime(
    "2024-07-29 12:00:00"
)  # Keep in mind thate only forecasts are made at intervals of 6 hours
upperbound = 48
upperbound_time = time + dt.timedelta(hours=upperbound)
current_forecast = forecasts[
    (forecasts.date == time) & (forecasts.date_of_forecast <= upperbound_time)
]

current_forecast = current_forecast.groupby(["region", "date_of_forecast"])[
    "ensembles"
].apply(lambda x: [item for sublist in x for item in sublist])

time_lb = 2
time_ub = 8

start_time = time + dt.timedelta(hours=time_lb)
end_time = time + dt.timedelta(hours=time_ub)

current_forecast_window = current_forecast.loc[
    (current_forecast.index.get_level_values("date_of_forecast") >= start_time)
    & (current_forecast.index.get_level_values("date_of_forecast") <= end_time)
]
print(current_forecast_window)

rain_threshold = 1
confidence = 0.9

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

quantile_values_by_region = {}

for region_name in current_forecast_window.index.get_level_values("region").unique():
    region_forecast_series = current_forecast_window.loc[region_name]
    quantile_values_by_region[region_name] = []

    for forecast_time, ensemble_values in region_forecast_series.items():
        ensemble_array = np.array(ensemble_values)

        if len(ensemble_array) == 0 or np.all(ensemble_array == 0):
            quantile_values_by_region[region_name].append(0.0)
            continue

        sorted_values = np.sort(ensemble_array)
        empirical_probabilities = np.linspace(0, 1, len(sorted_values))

        # Interpolate inverse CDF (quantile function)
        quantile_function = interp1d(
            empirical_probabilities,
            sorted_values,
            bounds_error=False,
            fill_value=(sorted_values[0], sorted_values[-1]),
        )

        # Append the rainfall value at the given confidence level
        # Here you calculate that with a certain confidence level, there will less rain that.
        # Thus: With 90% confidence, the rainfall will be less than X mm.
        quantile_values_by_region[region_name].append(
            float(quantile_function(confidence))
        )

ES_predicted = np.mean(quantile_values_by_region["ES"]) > rain_threshold

RZ_totals = [
    np.mean(quantile_values_by_region[region]) for region in ["RZ1", "RZ2", "GE"]
]
RZ_predicted = any(total > 3 for total in RZ_totals) or np.mean(RZ_totals) > 1


import matplotlib.pyplot as plt

# Plot individual region quantiles over time
for region_name, quantiles in quantile_values_by_region.items():
    plt.figure(figsize=(10, 4))
    plt.plot(
        quantiles,
        marker="o",
        linestyle="-",
        label=f"{region_name} ({confidence*100:.0f}th percentile)",
    )
