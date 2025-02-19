import netCDF4 as nc
import os

path = rf"data\precipitation\zip_raw_harmonie"

files = [
    f
    for f in os.listdir(path)
    if (os.path.isfile(os.path.join(path, f)) and f.endswith(".nc"))
]

regions = {
    "ES": [5.417, 5.533, 51.106, 51.486],
    "GE": [5.533, 5.641, 51.405, 51.451],
    "RZ1": [5.407, 5.486, 51.330, 51.406],
    "RZ2": [5.302, 5.409, 51.281, 51.344],
}

for file in files:
    with nc.Dataset(os.path.join(path, file), mode="r") as f:
        print(f.variables.keys())
    break
