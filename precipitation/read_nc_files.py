import netCDF4 as nc
import xarray as xr
import os
import re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

path = rf"data\precipitation\zip_raw_harmonie"
output_file = "forecast_data.csv"
output_path = rf"data\precipitation\csv_forecasts"

forecast_horizon = [
    "6",
    "12",
    "18",
    "24",
    "30",
    "36",
    "42",
    "48",
]  # Which forcast hours to select. Single digit hours should be written as 2 digit number; 0X

LON_BUFFER = 0.12 / 2
LAT_BUFFER = 0.07 / 2

files = [
    f
    for f in os.listdir(path)
    if (
        os.path.isfile(os.path.join(path, f))
        and f.endswith(".nc")
        and any(f"_0{horizon}.nc" in f for horizon in forecast_horizon)
    )
]

regions = {
    "ES": [5.417, 5.533, 51.106, 51.486],
    "GE": [5.533, 5.641, 51.405, 51.451],
    "RZ1": [5.407, 5.486, 51.330, 51.406],
    "RZ2": [5.302, 5.409, 51.281, 51.344],
}


def extract_timestamp(filename):
    pattern = re.compile(r"_(\d{8})(\d{2})_\d{3}_(\d{3})\.nc")
    match = pattern.search(filename)
    if match:
        date_str, hour_str, prediction_delta = match.groups()

        # Parse date and hour
        date_part = datetime.strptime(date_str, "%Y%m%d")
        hour_part = int(hour_str)  # Hours in HHH format
        delta_part = int(prediction_delta)

        # Create the timestamp
        timestamp = date_part.replace(hour=hour_part)
        timestamp_delta = timestamp + timedelta(hours=delta_part)
        return timestamp, timestamp_delta

    return None


data = []
for file in files:
    datetime_of_prediction, datetime_of_forecast = extract_timestamp(file)
    with xr.open_dataset(
        filename_or_obj=os.path.join(path, file), engine="netcdf4"
    ) as f:
        for region_name, (min_lon, max_lon, min_lat, max_lat) in regions.items():
            lon_mask = (min_lon - LON_BUFFER <= f.lon.values) & (
                f.lon.values <= max_lon + LON_BUFFER
            )  # take one extra grid cell
            lat_mask = (min_lat - LAT_BUFFER <= f.lat.values) & (
                f.lat.values <= max_lat + LAT_BUFFER
            )
            location_prediction_multi_grid = f.Precipitation[:, :, lat_mask, lon_mask]
            location_prediction_grid_mean = location_prediction_multi_grid.mean(
                dim=["lat", "lon"]
            )
            ensembles_predictions = xr.DataArray(
                location_prediction_grid_mean
            ).values.flatten()

            if ensembles_predictions.size != 50:
                ensembles_predictions = np.pad(
                    ensembles_predictions,
                    (0, 50 - ensembles_predictions.size),
                    mode="constant",
                    constant_values=np.nan,
                )

            data.append(
                (
                    region_name,
                    datetime_of_prediction,
                    datetime_of_forecast,
                    ensembles_predictions,
                )
            )

columns = ["region", "date", "date_of_forecast", "ensembles"]
df = pd.DataFrame(data, columns=columns)


# Check if the CSV file already exists
file_exists = os.path.isfile(os.path.join(output_path, output_file))

# If the file exists, append the new data
if file_exists:
    df_existing = pd.read_csv(
        os.path.join(output_path, output_file), index_col="timestamp", parse_dates=True
    )
    df_existing.loc[df.index] = df
    df_existing.to_csv(os.path.join(output_path, output_file))
else:
    # Otherwise, create the file and save the data
    df.to_csv(os.path.join(output_path, output_file))
print(f"Data saved to {output_file}")
