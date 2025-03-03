import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

week = pd.read_csv(rf"data\SWMM\week_pattern.csv", skiprows=1, index_col=0)
week.index = pd.to_datetime(week.index, format="%H:%M")
print(week.resample("H").mean())

weekend = pd.read_csv(rf"data\SWMM\weekend_pattern.csv", skiprows=1, index_col=0)
weekend.index = pd.to_datetime(weekend.index, format="%H:%M")
print(weekend.resample("H").mean())


swmm_pattern = [
    0.7847,
    0.71998333,
    0.68296667,
    0.66506667,
    0.70723333,
    0.8107,
    0.93276667,
    1.04196667,
    1.1301,
    1.17146667,
    1.1967,
    1.19685,
    1.17073333,
    1.1474,
    1.12601667,
    1.11725,
    1.13,
    1.14623333,
    1.14261667,
    1.11025,
    1.06796667,
    1.01133333,
    0.93668333,
    0.85298333,
]

west_pattern3 = [
    0.2,
    0.2,
    0.2,
    0.3,
    0.6,
    0.9,
    1.1,
    1.3,
    1.3,
    1.4,
    1.6,
    1.8,
    2,
    1.8,
    1.4,
    1.2,
    1.1,
    1.1,
    1.1,
    1.2,
    1,
    0.7,
    0.3,
    0.2,
]
t = np.arange(0, 24)

plt.plot(swmm_pattern, label="swmm")
plt.plot(west_pattern3, label="west")

west_pattern_1 = [
    0.1,
    0.1,
    0.1,
    0.1,
    0.6,
    1.6,
    1.9,
    1.3,
    1.1,
    0.8,
    1,
    1.9,
    3,
    2.5,
    1.6,
    0.8,
    0.7,
    1.3,
    1.8,
    1.5,
    0.4,
    0.1,
    0.1,
    0.1,
]


"""
############ Evaporation average per month:timestamp
2024-01-31    0.468640
2024-02-29    0.827869
2024-03-31    1.441665
2024-04-30    2.157225
2024-05-31    2.774805
2024-06-30    3.129237
2024-07-31    3.121963
2024-08-31    2.749193
2024-09-30    2.122777
2024-10-31    1.407267
2024-11-30    0.795271
2024-12-31    0.455212
Freq: M, Name: .c_106.runoff.In_1(Evaporation), dtype: float64

But is highyl oscialating, in amrch between 0,4 at night till 4 mm/d during day
"""
