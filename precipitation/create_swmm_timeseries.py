import pandas as pd
import os

# Read CSV file
data = pd.read_csv(
    r"data\precipitation\selected_euradclim\5_min_mean_precipitation_data.csv",
    index_col="timestamp",
    parse_dates=True,
)

# Get all locations (columns)
locations = list(data.keys())

# Directory to save output files
output_dir = "data/precipitation/swmm_rain_data"
os.makedirs(output_dir, exist_ok=True)

MONTH = 12


# Function to write SWMM rain gauge format
def write_swmm_file(file_path, location, time_series):
    with open(file_path, "w") as f:
        # Rain gauge definition
        f.write(
            f"; Rain Gage Data, 5-min rainfall expressed as Cumulative Depth in MM\n"
        )

        for timestamp, value in time_series.items():
            year = timestamp.year
            month = timestamp.month
            day = timestamp.day
            hour = timestamp.hour
            minute = timestamp.minute
            f.write(f"{location} {year} {month} {day} {hour} {minute} {value:.2f}\n")


# Loop through each location
for location in locations:
    # Extract yearly data
    year_data = data[location].dropna()  # Remove NaN values
    year_file = os.path.join(output_dir, f"{location}_yearly.dat")
    write_swmm_file(year_file, location, year_data)

    month_data = data.loc[data.index.month == MONTH, location].dropna()
    if not month_data.empty:  # Ensure there's data before saving
        month_file = os.path.join(output_dir, f"{location}_month_{MONTH}.dat")
        write_swmm_file(month_file, location, month_data)

print("SWMM rain gauge files have been successfully created!")
