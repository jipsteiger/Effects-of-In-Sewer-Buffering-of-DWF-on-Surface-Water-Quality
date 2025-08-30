from simulation import Simulation
from realtimecontrol import RealTimeControl
from postprocess import PostProcess
import datetime as dt
import pandas as pd

"""
This file makes for all possible scenarios (RTC, RTC w/ Ensemble forecasts, no RTC, no RTC w/o rain, and no RTC w/ constant outflow and w/o rain)
"""

all_results = {}

MODEL_NAME = "model_jip"
SUFFIX = "RTC_new_rain"
simulation = RealTimeControl(
    model_path=rf"data\SWMM\{MODEL_NAME}.inp",
    step_size=300,
    report_start=dt.datetime(year=2024, month=7, day=1),
    start_time=dt.datetime(year=2024, month=7, day=1),
    end_time=dt.datetime(year=2024, month=8, day=1),
    virtual_pump_max=10,
)
simulation.start_simulation()

timesteps, ES_states, RZ_states = simulation.get_state()
all_results[f"ES_{SUFFIX}"] = ES_states
all_results[f"RZ_{SUFFIX}"] = RZ_states


postprocess = PostProcess(model_name=MODEL_NAME)
postprocess.create_outfall_txt_concentrate(suffix=SUFFIX, specific_version="RTC")

SUFFIX = "Ensemble_RTC_new_rain"
MODEL_NAME = "model_jip_ENSEMBLE"
simulation = RealTimeControl(
    model_path=rf"data\SWMM\{MODEL_NAME}.inp",
    step_size=300,
    report_start=dt.datetime(year=2024, month=7, day=1),
    start_time=dt.datetime(year=2024, month=7, day=1),
    end_time=dt.datetime(year=2024, month=8, day=1),
    virtual_pump_max=10,
    use_ensemble_forecast=True,
    ES_threshold=0.75,
    RZ_threshold=1,
    ES_certainty_threshold=0.7,
    RZ_certainty_threshold=0.925,
)
simulation.start_simulation()
timesteps, ES_states, RZ_states = simulation.get_state()
all_results[f"ES_{SUFFIX}"] = ES_states
all_results[f"RZ_{SUFFIX}"] = RZ_states

postprocess = PostProcess(model_name=MODEL_NAME)
postprocess.create_outfall_txt_concentrate(suffix=SUFFIX, specific_version="RTC")

SUFFIX = "No_RTC_mixed_sim"
MODEL_NAME = "model_jip_no_rtc"
simulation = Simulation(
    model_path=rf"data\SWMM\{MODEL_NAME}.inp",
    step_size=300,
    report_start=dt.datetime(year=2024, month=7, day=1),
    start_time=dt.datetime(year=2024, month=7, day=1),
    end_time=dt.datetime(year=2024, month=8, day=1),
    virtual_pump_max=10,
)
simulation.start_simulation()

postprocess = PostProcess(model_name=MODEL_NAME)
postprocess.create_outfall_txt_concentrate(suffix=SUFFIX, specific_version="no_RTC")


SUFFIX = "No_RTC_no_rain"
MODEL_NAME = "model_jip_no_rtc_no_rain"
simulation = Simulation(
    model_path=rf"data\SWMM\{MODEL_NAME}.inp",
    step_size=300,
    report_start=dt.datetime(year=2023, month=4, day=15),
    start_time=dt.datetime(year=2023, month=4, day=15),
    end_time=dt.datetime(year=2023, month=10, day=16),
    virtual_pump_max=10,
)
simulation.start_simulation()


postprocess = PostProcess(model_name=MODEL_NAME)
postprocess.create_outfall_txt_concentrate(suffix=SUFFIX, specific_version="no_RTC")

SUFFIX = "No_RTC_no_rain_constant"
MODEL_NAME = "model_jip_no_rtc_no_rain_constant"
simulation = Simulation(
    model_path=rf"data\SWMM\{MODEL_NAME}.inp",
    step_size=300,
    report_start=dt.datetime(year=2023, month=4, day=15),
    start_time=dt.datetime(year=2023, month=4, day=15),
    end_time=dt.datetime(year=2023, month=10, day=16),
    virtual_pump_max=10,
    constant_outflow=True,
)
simulation.start_simulation()
timesteps, ES_states, RZ_states = simulation.get_state()
all_results[f"ES_{SUFFIX}"] = ES_states
all_results[f"RZ_{SUFFIX}"] = RZ_states

postprocess = PostProcess(model_name=MODEL_NAME)
postprocess.create_outfall_txt_concentrate(suffix=SUFFIX, specific_version="no_RTC")


df = pd.DataFrame(all_results, index=timesteps)
df.to_csv("simulation_states_systems.csv")


####################
