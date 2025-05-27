import pandas as pd
import matplotlib.pyplot as plt
import swmm_api as sa

def dry_weather_flow():
    swmm = sa.read_out_file(rf'data\SWMM\model_jip_no_rtc.out').to_frame()
    west = pd.read_csv(rf'data\WEST\Model_Dommel_Full\westcompare.out.txt',        
            delimiter="\t",
            header=0,
            index_col=0,
            low_memory=False,
        ).iloc[1:, :]
    start_date = pd.Timestamp("2024-01-01")
    west["timestamp"] = start_date + pd.to_timedelta(
        west.index.astype(float), unit="D"
    )
    
    ES_Q_swmm = swmm.link.P_eindhoven_out.flow
    RZ_Q_swmm = swmm.link.P_riool_zuid_out.flow
    ES_Q_west = west['.pipe_ES.Q_Out']
    RZ_Q_west = west['.pipe_RZ.Q_Out']
    
    ES_Q_swmm = ES_Q_swmm.loc['2024-09-20':'2024-09-23']
    RZ_Q_swmm = RZ_Q_swmm.loc['2024-09-20':'2024-09-23']
    ES_Q_west = ES_Q_west.loc['2024-09-20':'2024-09-23']
    RZ_Q_west = RZ_Q_west.loc['2024-09-20':'2024-09-23']
    
    plt.figure()
    ES_Q_swmm.plot()
    ES_Q_west.plot()
    plt.figure()
    RZ_Q_swmm.plot()
    RZ_Q_west.plot()