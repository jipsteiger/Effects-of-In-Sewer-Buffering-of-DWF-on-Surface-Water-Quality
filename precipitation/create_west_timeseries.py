import pandas as pd

# Read the data and select specific columns using 'usecols'
# Read the data, assuming the first row is the header
df = pd.read_csv(
    r"data\precipitation\csv_selected_area_euradclim\2024_5_min_precipitation_data.csv",
    parse_dates=["timestamp"],
)
df.set_index("timestamp", inplace=True)
df = df.resample("h").sum()
# Reset index to get timestamp back as column (optional)
df.reset_index(inplace=True)

# Convert time delta to days (relative to the first timestamp)
df["t_days"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds() / (
    24 * 3600
)

# Prepare the required output DataFrame
output_df = df[["t_days", "ES"]].rename(columns={"t_days": "#.t", "ES": ".in_1"})

# Save to file with headers as shown
with open(
    rf"data\precipitation\dat_swmm_rain_data\es_year_24_WEST_format.txt", "w"
) as f:
    f.write("#.t\t.in_1\n")
    f.write("#d\tmm\n")
    output_df.to_csv(f, sep="\t", index=False, header=False, float_format="%.7f")
