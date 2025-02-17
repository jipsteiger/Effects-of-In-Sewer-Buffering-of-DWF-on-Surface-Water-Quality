import h5py
import os
import zipfile
import shutil  # To move files
import pandas as pd
import numpy as np
from pyproj import Proj, Transformer
import netCDF4 as nc
import re

path = r"data\precipitation\raw_euradclim"

files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

print(files)

extract_to = r"data\precipitation\euradclim_5min_cdf"


def extract_zip():
    for file in files:
        with zipfile.ZipFile(os.path.join(path, file), "r") as zip_ref:
            # Get all files in ZIP archive
            all_files = zip_ref.namelist()

            # Find the deepest nested directory
            deepest_folders = {}
            for f in all_files:
                if f.endswith("/"):  # It's a folder
                    depth = f.count("/")
                    deepest_folders[f] = depth

            # Get the deepest folder path
            deepest_folder = max(deepest_folders, key=deepest_folders.get, default="")

            # Extract only HDF5 files
            extracted_files = []
            temp_extract_path = os.path.join(
                extract_to, "temp_extracted"
            )  # Temporary folder
            os.makedirs(temp_extract_path, exist_ok=True)

            for f in all_files:
                if (f.endswith(".h5")) or (f.endswith(".nc")):
                    extracted_path = zip_ref.extract(f, temp_extract_path)

                    # Move extracted file to the main folder (flatten structure)
                    flat_path = os.path.join(
                        extract_to, os.path.basename(extracted_path)
                    )
                    shutil.move(
                        extracted_path, flat_path
                    )  # Move and overwrite if needed
                    extracted_files.append(flat_path)

            # Remove temporary extracted folder (cleans up nested structure)
            shutil.rmtree(temp_extract_path, ignore_errors=True)


path = r"data\precipitation\euradclim_5min"

h5_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
write_to = r"data\precipitation\selected_euradclim"

# code / coordinates from https://github.com/overeem11/EURADCLIM-tools

DatasetNr = "/image1"
ATTR_NAME = DatasetNr + "/what"
DATAFIELD_NAME = DatasetNr + "/data1/data"
Grid = np.array(
    pd.read_csv(
        r"precipitation\CoordinatesHDF5ODIMWGS84.dat",
        delimiter=" ",
        dtype="float",
        header=None,
    )
)
Xcoor = Grid[:, 0]
Ycoor = Grid[:, 1]

regions = {
    "ES": [5.417, 5.533, 51.106, 51.486],
    "GE": [5.533, 5.641, 51.405, 51.451],
    "RZ1": [5.407, 5.486, 51.330, 51.406],
    "RZ2": [5.302, 5.409, 51.281, 51.344],
}

# Initialize an empty dictionary to store mean precipitation for each region
mean_precip_data = {region: [] for region in regions}
timestamps = []  # List to store timestamps

# Define output file
output_file = "5_min_mean_precipitation_data.csv"
output_path = rf"data\precipitation\selected_euradclim"

# Check if the CSV file already exists
file_exists = os.path.isfile(os.path.join(output_path, output_file))


def hourly():
    for file in h5_files:
        with h5py.File(os.path.join(path, file), "r") as f:
            # Read metadata
            Ncols = int(f["/where"].attrs["xsize"])
            Nrows = int(f["/where"].attrs["ysize"])
            zscale = f[ATTR_NAME].attrs["gain"]
            zoffset = f[ATTR_NAME].attrs["offset"]
            nodata = f[ATTR_NAME].attrs["nodata"]
            undetect = f[ATTR_NAME].attrs["undetect"]

            # Read dataset and scale values
            dset = f[DATAFIELD_NAME][:]
            RArray = zoffset + zscale * dset

            # Load and reshape longitude & latitude coordinates
            Xcoor = np.array(Xcoor).reshape((Nrows, Ncols))
            Ycoor = np.array(Ycoor).reshape((Nrows, Ncols))

            # Convert timestamp
            timestamp_str = file.split("_")[-1].split(".")[0]
            timestamp = pd.to_datetime(timestamp_str, format="%Y%m%d%H%M")

            # Process each region
            for region_name, (min_lon, max_lon, min_lat, max_lat) in regions.items():
                # Create mask for the bounding box
                lon_mask = (Xcoor >= min_lon) & (Xcoor <= max_lon)
                lat_mask = (Ycoor >= min_lat) & (Ycoor <= max_lat)
                region_mask = lon_mask & lat_mask  # Combine masks

                # Extract subset
                subset_RArray = RArray[region_mask]

                # Calculate mean precipitation
                mean_precipitation = round(subset_RArray.mean(), 3)

                # Append mean precipitation to the corresponding region's list
                mean_precip_data[region_name].append(mean_precipitation)

            # Store the timestamp for each file
            timestamps.append(timestamp)


regions = {
    "ES": [5.417, 5.533, 51.106, 51.486],
    "GE": [5.533, 5.641, 51.405, 51.451],
    "RZ1": [5.407, 5.486, 51.330, 51.406],
    "RZ2": [5.302, 5.409, 51.281, 51.344],
}

path = r"data\precipitation\euradclim_5min"

h5_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
write_to = r"data\precipitation\selected_euradclim"

# Initialize an empty dictionary to store mean precipitation for each region
mean_precip_data = {region: [] for region in regions}
timestamps = []  # List to store timestamps

# Define output file
output_file = "5_min_mean_precipitation_data.csv"
output_path = rf"data\precipitation\selected_euradclim"

# Check if the CSV file already exists
file_exists = os.path.isfile(os.path.join(output_path, output_file))


def five_min_data():
    for i, file in enumerate(h5_files):
        with h5py.File(os.path.join(path, file), "r") as f:
            # Read geographic attributes
            Ncols = int(f["/geographic"].attrs["geo_number_columns"])
            Nrows = int(f["/geographic"].attrs["geo_number_rows"])
            pixel_size_x = float(f["/geographic"].attrs["geo_pixel_size_x"])
            pixel_size_y = float(f["/geographic"].attrs["geo_pixel_size_y"])
            product_corners = f["/geographic"].attrs["geo_product_corners"]
            LL_lon, UL_lon, UR_lon, LR_lon = (
                product_corners[0],
                product_corners[2],
                product_corners[4],
                product_corners[6],
            )
            LL_lat, UL_lat, UR_lat, LR_lat = (
                product_corners[1],
                product_corners[3],
                product_corners[5],
                product_corners[7],
            )

            # Read the calibration information for nodata and undetect values
            nodata = f["/image1/calibration"].attrs["calibration_missing_data"]
            undetect = f["/image1/calibration"].attrs["calibration_out_of_image"]

            # Read the image data
            image_data = f["/image1/image_data"][:]

            Xcoor = np.linspace(
                np.array([LL_lon, UL_lon, UR_lon, LR_lon]).min(),
                np.array([LL_lon, UL_lon, UR_lon, LR_lon]).max(),
                Ncols,
            )  # Longitude from west to east
            Ycoor = np.linspace(
                np.array([LL_lat, UL_lat, UR_lat, LR_lat]).min(),
                np.array([LL_lat, UL_lat, UR_lat, LR_lat]).max(),
                Nrows,
            )  # Latitude from north to south

            # Create a meshgrid for the coordinates (longitude, latitude)
            grid = np.meshgrid(Xcoor, Ycoor[::-1])

            Xcoor_grid = grid[0].reshape((256, 256))
            Ycoor_grid = grid[1].reshape((256, 256))

            # Process the image data to calculate the mean precipitation for each region
            for region_name, (min_lon, max_lon, min_lat, max_lat) in regions.items():
                # Create a mask for longitude and latitude within the region's bounding box
                lon_mask = (Xcoor_grid >= min_lon) & (Xcoor_grid <= max_lon)
                lat_mask = (Ycoor_grid >= min_lat) & (Ycoor_grid <= max_lat)
                region_mask = lon_mask & lat_mask  # Combine both masks

                # Extract the subset of the image data for this region
                subset_RArray = image_data[region_mask]

                # Mask out the nodata and undetect values from the subset
                subset_RArray = subset_RArray[
                    (subset_RArray != nodata) & (subset_RArray != undetect)
                ]

                # Calculate mean precipitation (or other statistics) for the region, skipping invalid data
                if len(subset_RArray) > 0:  # Only calculate if there is valid data
                    mean_precipitation = round(subset_RArray.mean(), 3) * 0.01
                else:
                    mean_precipitation = -0.0001  # Handle case with no valid data

                # Store the result
                mean_precip_data[region_name].append(mean_precipitation)

            # Example: store the timestamp or any other metadata if available
            match = re.search(r"(\d{12})", file)

            if match:
                datetime_str = match.group(1)
                timestamp = pd.to_datetime(datetime_str, format="%Y%m%d%H%M")
                timestamps.append(timestamp)
            else:
                timestamps.append(pd.to_datetime("190001010000", format="%Y%m%d%H%M"))


# Output the final results
print("Mean precipitation data:", mean_precip_data)
print("Timestamps:", timestamps)


# Create DataFrame from the collected data
df = pd.DataFrame(mean_precip_data)

# Add timestamps as the index
df["timestamp"] = timestamps
df.set_index("timestamp", inplace=True)

df[df < 0] = 0

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

df = pd.read_csv(
    rf"data\precipitation\selected_euradclim\5_min_mean_precipitation_data.csv",
    index_col="timestamp",
    parse_dates=True,
)


import matplotlib.pyplot as plt

plt.figure(figsize=(30, 10))
df_day = df.resample("D").sum()
df_day.plot(figsize=(30, 10))
plt.ylabel("precipitation [mm]")
