import swmm_api as sa
import pandas as pd

output = sa.SwmmOutput(rf"data\SWMM\model_jip_geen_regen.out").to_frame()
df = output.loc["2023-07-01":"2023-08-01"]

Q_out_ES_avg = df.loc["2023-07-01"].link.P_eindhoven_out.flow.mean()
Q_out_RZ_avg = df.loc["2023-07-01"].link.P_riool_zuid_out.flow.mean()
Q_out_aalst_avg = df.loc["2023-07-01"].node.Nod_127.total_inflow.mean()
Q_out_mierlo_avg = df.loc["2023-07-01"].link["Con_1119.1"].flow.mean()
Q_out_geldrop_avg = df.loc["2023-07-01"].node.J1.total_inflow.mean()

Q_out_ES_max = 3.888
Q_out_RZ_max = 4.7222

area_eindhoven = sum([1918.8, 3, 1.6, 3.2, 3.7, 10, 3.4, 6.2, 26.4, 33.9, 35.6])
area_geldrop = sum([5, 255.1, 5.8, 134.9, 15.8, 5.3, 1])
area_mierlo = sum([50.5, 22, 21.5, 3.9, 5.3, 3.9])
area_aalst = sum(
    [
        205,
        101.5,
        308,
        4.3,
        5,
        1,
        22.8,
        76.8,
        11.2,
        18.3,
        29.4,
        68,
        28.9,
        4.6,
        1,
        52.2,
        10.3,
        4.9,
        393,
        6.7,
        11.3,
        12.9,
        6.1,
        28.5,
        77.2,
        9,
        11.3,
        103.7,
        3.8,
        26.2,
    ]
)

storage_ES = 165_000
storage_TP = 21_407.57  # Part of storage only that gets used for DWF stuff
storage_geldrop = sum([8_600, 27_800])
storage_mierlo = 8_632
storage_aalst = sum([143_269.80, 16739, 1700, 26000, 7960])
