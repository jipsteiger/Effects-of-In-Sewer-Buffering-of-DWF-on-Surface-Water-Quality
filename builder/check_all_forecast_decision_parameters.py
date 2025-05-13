from realtimecontrol import RealTimeControl
import datetime as dt
import swmm_api as sa
import pandas as pd
import os
import numpy as np
import itertools
import time
import shutil

MODEL_NAME = "model_jip"
SUFFIX = "RTC"

ES_thresholds = np.arange(0.5, 3, 0.25)
RZ_thresholds = np.arange(0.5, 5, 0.5)
certainty_thresholds = np.arange(0.725, 0.95, 0.025)


# ES_thresholds = np.arange(0.5, 2.5, 1)
# RZ_thresholds = np.arange(0.5, 2.5, 1)
# certainty_thresholds = np.arange(0.9, 0.95, 0.05)

iterations = len(ES_thresholds) * len(certainty_thresholds)
minutes = iterations * 4
hours = minutes / 60
print(hours)
iterations = len(RZ_thresholds) * len(certainty_thresholds)
minutes = iterations * 4
hours = minutes / 60
print(hours)


output_folder = os.path.join("data", "SWMM", "outfile_saved")
os.makedirs(output_folder, exist_ok=True)


start = time.time()

all_results = {}  # Dictionary to store the results

# Loop for ES
for ES_threshold, certainty_threshold in itertools.product(
    ES_thresholds, certainty_thresholds
):
    simulation = RealTimeControl(
        model_path=rf"data\SWMM\{MODEL_NAME}.inp",
        step_size=300,
        report_start=dt.datetime(year=2024, month=7, day=1),
        start_time=dt.datetime(year=2024, month=7, day=1),
        end_time=dt.datetime(year=2024, month=7, day=31),
        virtual_pump_max=10,
        use_ensemble_forecast=True,
        constant_outflow=False,
        ES_threshold=ES_threshold,
        ES_certainty_threshold=certainty_threshold,
    )
    simulation.start_simulation()
    timesteps, ES_state, RZ_states = simulation.get_state()

    col_name = f"ES_{ES_threshold}_cert_{certainty_threshold}"
    all_results[col_name] = ES_state

    # Create a unique name based on the simulation type and threshold
    file_tag = (
        f"ES_{ES_threshold}_mm_{certainty_threshold}_cert"  # or RZ_... in RZ loop
    )
    source_out_file = os.path.join("data", "SWMM", "model_jip.out")
    destination_out_file = os.path.join(output_folder, f"{file_tag}.out")

    # Copy the output file
    shutil.copy(source_out_file, destination_out_file)

# Loop for RZ
for RZ_threshold, certainty_threshold in itertools.product(
    RZ_thresholds, certainty_thresholds
):
    simulation = RealTimeControl(
        model_path=rf"data\SWMM\{MODEL_NAME}.inp",
        step_size=300,
        report_start=dt.datetime(year=2024, month=7, day=1),
        start_time=dt.datetime(year=2024, month=7, day=1),
        end_time=dt.datetime(year=2024, month=7, day=31),
        virtual_pump_max=10,
        use_ensemble_forecast=True,
        constant_outflow=False,
        RZ_threshold=RZ_threshold,
        RZ_certainty_threshold=certainty_threshold,
    )
    simulation.start_simulation()
    timesteps, ES_state, RZ_states = simulation.get_state()

    col_name = f"RZ_{RZ_threshold}_cert_{certainty_threshold}"
    all_results[col_name] = RZ_states

    # Create a unique name based on the simulation type and threshold
    file_tag = f"RZ_{RZ_threshold}_mm_{certainty_threshold}_cert"
    source_out_file = os.path.join("data", "SWMM", "model_jip.out")
    destination_out_file = os.path.join(output_folder, f"{file_tag}.out")

    # Copy the output file
    shutil.copy(source_out_file, destination_out_file)


# Combine all results into a DataFrame
df = pd.DataFrame(all_results, index=timesteps)

# Save to CSV
df.to_csv("simulation_states.csv")

end = time.time()
print(f"Total duration = {end-start}")


simulation = RealTimeControl(
    model_path=rf"data\SWMM\{MODEL_NAME}.inp",
    step_size=300,
    report_start=dt.datetime(year=2024, month=7, day=1),
    start_time=dt.datetime(year=2024, month=7, day=1),
    end_time=dt.datetime(year=2024, month=7, day=31),
    virtual_pump_max=10,
)
simulation.start_simulation()
timesteps, ES_states, RZ_states = simulation.get_state()

col_name = f"Base_perfect_prediction"
all_results[f"ES_{col_name}"] = ES_states
all_results[f"RZ_{col_name}"] = RZ_states

# Create a unique name based on the simulation type and threshold
file_tag = f"Base_predition"
source_out_file = os.path.join("data", "SWMM", "model_jip.out")
destination_out_file = os.path.join(output_folder, f"{file_tag}.out")

# Copy the output file
shutil.copy(source_out_file, destination_out_file)


# Combine all results into a DataFrame
df = pd.DataFrame(all_results, index=timesteps)

# Save to CSV
df.to_csv("simulation_states.csv")
