import pandas as pd
import matplotlib.pyplot as plt

start_date = pd.Timestamp("2024-01-01")
constant = pd.read_csv(
    rf"output_swmm\05-29_16-26_out_ES_No_RTC_no_rain_constant.csv",
    decimal=",",
    delimiter=";",
    index_col=0,
)
constant["timestamp"] = start_date + pd.to_timedelta(
    constant.index.astype(float), unit="D"
)
constant.set_index("timestamp", inplace=True)
constant.index = constant.index.round("15min")
dry = pd.read_csv(
    rf"output_swmm\05-29_16-24_out_ES_No_RTC_no_rain.csv",
    decimal=",",
    delimiter=";",
    index_col=0,
)
dry["timestamp"] = start_date + pd.to_timedelta(dry.index.astype(float), unit="D")
dry.set_index("timestamp", inplace=True)
dry.index = dry.index.round("15min")

scenarios = {
    "Dry weather flow only": dry,
    "Constant dry weather flow only": constant,
}

# Define your time window
start_date = pd.Timestamp("2024-04-15")
end_date = pd.Timestamp("2024-04-25")

# Column to plot (replace with actual column name)
column_to_plot = "TotalOutflow"

for key in list(dry.keys())[0:2]:
    if not ("FD" in key or "Q_out" in key):
        # Filter by time window and store filtered versions
        filtered_scenarios = {
            key: df.loc[start_date:end_date] for key, df in scenarios.items()
        }

        # Now plot all filtered scenarios together
        plt.figure(figsize=(14, 6))

        for label, df in filtered_scenarios.items():
            if key in df.columns:
                plt.plot(df.index, df[key], label=label)
            else:
                print(f"Column '{key}' not found in scenario '{label}'")

        plt.title(f"{key} from {start_date.date()} to {end_date.date()}")
        plt.xlabel("Timestamp")
        plt.ylabel(key)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

import pandas as pd
import matplotlib.pyplot as plt

start_date = pd.Timestamp("2024-01-01")
constant = pd.read_csv(
    rf"output_swmm\05-29_16-30_out_ES_No_RTC_no_rain_constant.csv",
    decimal=",",
    delimiter=";",
    index_col=0,
)
constant["timestamp"] = start_date + pd.to_timedelta(
    constant.index.astype(float), unit="D"
)
constant.set_index("timestamp", inplace=True)
constant.index = constant.index.round("15min")
dry = pd.read_csv(
    rf"output_swmm\05-29_16-29_out_ES_No_RTC_no_rain.csv",
    decimal=",",
    delimiter=";",
    index_col=0,
)
dry["timestamp"] = start_date + pd.to_timedelta(dry.index.astype(float), unit="D")
dry.set_index("timestamp", inplace=True)
dry.index = dry.index.round("15min")

scenarios = {
    "Dry weather flow only": dry,
    "Constant dry weather flow only": constant,
}

# Define your time window
start_date = pd.Timestamp("2024-04-15")
end_date = pd.Timestamp("2024-04-25")

# Column to plot (replace with actual column name)
column_to_plot = "TotalOutflow"

for key in list(dry.keys())[0:2]:
    if not ("FD" in key or "Q_out" in key):
        # Filter by time window and store filtered versions
        filtered_scenarios = {
            key: df.loc[start_date:end_date] for key, df in scenarios.items()
        }

        # Now plot all filtered scenarios together
        plt.figure(figsize=(14, 6))

        for label, df in filtered_scenarios.items():
            if key in df.columns:
                plt.plot(df.index, df[key], label=label)
            else:
                print(f"Column '{key}' not found in scenario '{label}'")

        plt.title(f"{key} from {start_date.date()} to {end_date.date()}")
        plt.xlabel("Timestamp")
        plt.ylabel(key)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
