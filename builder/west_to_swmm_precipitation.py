import swmm_api as sa
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import os as os
from math import isnan

path = r"data\WEST\Model_Dommel_Full"

precipitation_files = [
    f
    for f in os.listdir(path)
    if os.path.isfile(os.path.join(path, f)) and ".Dynamic.in.txt" in f
][1:]
write_to = r"data\precipitation\west_precipitation"


def write_swmm_file(file_path, location, time_series):
    with open(file_path, "w") as f:
        # Rain gauge definition
        f.write(
            f"; Rain Gage Data, 5-min rainfall expressed as Cumulative Depth in MM\n"
        )

        for timestamp, value in time_series.items():
            if isnan(value):
                value = 0
            year = timestamp.year
            month = timestamp.month
            day = timestamp.day
            hour = timestamp.hour
            minute = timestamp.minute
            f.write(f"{location} {year} {month} {day} {hour} {minute} {value:.6f}\n")


for f in precipitation_files:
    file = pd.read_csv(
        os.path.join(path, f),
        delimiter="\t",
        header=0,
        index_col=0,
    ).iloc[1:, :]
    start_date = pd.Timestamp("2024-01-01")
    file["timestamp"] = start_date + pd.to_timedelta(file.index.astype(float), unit="D")
    file["timestamp"] = file["timestamp"].dt.round("15min")
    file.set_index("timestamp", inplace=True)
    file_5m = file.resample("5T").ffill()
    file_5m.iloc[:, 0] = file_5m.iloc[:, 0].astype(float) / 12

    file_name = f"location_{f[9:11]}.dat"
    write_swmm_file(os.path.join(write_to, file_name), f[9:11], file_5m.iloc[:, 0])


x = 154836.120
y = 385897.610
for f in precipitation_files:
    number = f[9:11]

    # print(f'rain_gage_{number}\tVOLUME\t0:05\t1.0\tFILE\t'
    #     f'"D:\\Jip\\Documents\\Uni\\Master\\Thesis\\data\\precipitation\\west_precipitation\\location_{number}.dat"\t'
    #     f'{number}\tMM')

    y -= 500
    # print(f"rain_gage_{number}\t{x}\t{y} ")
