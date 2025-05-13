import datetime as dt
import pandas as pd
import numpy as np
import swmm_api as sa
import os as os
from storage import RZ_storage
import re
import time

states = pd.read_csv(rf"simulation_states.csv", index_col=0, parse_dates=True)
filenames = os.listdir(rf"data/SWMM/outfile_saved")
path = os.path.join("data", "SWMM", "outfile_saved")


def main():
    results = {
        "location": [],
        "rain_threshold": [],
        "certainty_threshold": [],
        "OF_1_outflow": [],
        "OF_1_FD": [],
        "OF_2_cso": [],
        "OF_3_margin_025": [],
        "OF_3_margin_05": [],
        "OF_3_margin_10": [],
        "OF_3_margin_15": [],
    }
    for filename in filenames:
        location, threshold, certainty = (
            get_location_rain_threshold_certainty_threshold(filename)
        )
        results["location"].append(location)
        results["rain_threshold"].append(threshold)
        results["certainty_threshold"].append(certainty)

        output = sa.read_out_file(os.path.join(path, filename)).to_frame()

        state = states.loc[:, find_collumn_name_from_filename(filename)]
        output, state = output.align(
            state, join="inner", axis=0
        )  # Align on the index (rows)

        outflow, FD, outflow_ideal = get_catchment_specific_flow_and_FD(
            output, location
        )
        OF_1_outflow, OF_1_FD = objective_function_1(outflow, FD, state, outflow_ideal)
        results["OF_1_outflow"].append(OF_1_outflow)
        results["OF_1_FD"].append(OF_1_FD)

        OF_2_cso = objective_function_2(output, location)
        results["OF_2_cso"].append(OF_2_cso)

        OF_3_margin_025 = objective_function_3(outflow, outflow_ideal, margin=1.025)
        OF_3_margin_05 = objective_function_3(outflow, outflow_ideal, margin=1.05)
        OF_3_margin_10 = objective_function_3(outflow, outflow_ideal, margin=1.1)
        OF_3_margin_15 = objective_function_3(outflow, outflow_ideal, margin=1.5)
        results["OF_3_margin_025"].append(OF_3_margin_025)
        results["OF_3_margin_05"].append(OF_3_margin_05)
        results["OF_3_margin_10"].append(OF_3_margin_10)
        results["OF_3_margin_15"].append(OF_3_margin_15)

        # One more for Loads

        # One more for WWTP effluent

    # Convert the results dictionary to a pandas DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.round(3)
    # Save the DataFrame as a CSV file
    results_df.to_csv("objective_function_values.csv", index=False)


def objective_function_1(outflow, FD, state, outflow_ideal):
    wanted_state_outflow = np.logical_or(
        state.values == "dwf", state.values == "transition"
    )
    wanted_state_FD = get_first_wwf_state(state)

    dwf_transition_outflow = outflow[wanted_state_outflow]
    initial_wwf_FD = FD[wanted_state_FD]

    OF_outflow = (dwf_transition_outflow - outflow_ideal) ** 2 / outflow_ideal

    return np.mean(OF_outflow.values), np.mean(initial_wwf_FD)


def objective_function_2(output, location):
    swmm_csos = {
        "ES": "cso_ES_1",
        "RZ": [
            "cso_gb_136",
            "cso_Geldrop",
            "cso_gb127",
            "cso_RZ",
            "cso_AALST",
            "cso_c_123",
            "cso_c_122",
            "cso_c_119_1",
            "cso_c_119_2",
            "cso_c_119_3",
            "cso_c_112",
            "cso_c_99",
        ],
    }
    csos = swmm_csos[location]
    # Check if 'csos' is a list or a string and sum accordingly
    if isinstance(csos, list):  # For RZ
        total_inflow = sum(
            output.node[cso]["total_inflow"].values.sum() for cso in csos
        )
    else:  # For ES (single CSO)
        total_inflow = output.node[csos]["total_inflow"].values.sum()

    return total_inflow


def objective_function_3(outflow, outflow_ideal, margin=1.1):
    dwf_outflow = outflow[outflow < outflow_ideal * margin]
    OF_outflow = (dwf_outflow - outflow_ideal) ** 2 / outflow_ideal
    return len(OF_outflow)


def get_first_wwf_state(state):
    # Identify the condition for 'wwf'
    wwf_condition = state.values == "wwf"

    # Create a mask for the first 'wwf' in a consecutive sequence
    first_wwf = np.zeros_like(wwf_condition, dtype=bool)

    # Mark the first occurrence of 'wwf' in the sequence
    for i in range(1, len(wwf_condition)):
        if wwf_condition[i] and not wwf_condition[i - 1]:
            first_wwf[i] = True

    # If the first element is 'wwf', mark it as True as well
    first_wwf[0] = wwf_condition[0]
    return first_wwf


def get_catchment_specific_flow_and_FD(output, location):
    if "ES" in location:
        outflow = output.link.P_eindhoven_out.flow
        outflow_ideal = 0.663
        FD = output.node.pipe_ES.volume / 11000
    if "RZ" in location:
        outflow = output.link.P_riool_zuid_out.flow
        storage = RZ_storage(7500)
        FD = storage.get_volume(output.link) / 7500
        outflow_ideal = 0.5218
    return outflow, FD, outflow_ideal


def find_collumn_name_from_filename(filename: str):
    # Remove .out extension
    base = os.path.splitext(filename)[0]

    # Replace '_mm_' with '_cert_'
    base = base.replace("_mm_", "_cert_")

    # Remove trailing '_cert' if it exists
    if base.endswith("_cert"):
        base = base[:-5]  # remove last 5 chars

    if "Base_prediction" in filename:
        base = filename[0:2] + "_base_prediction"

    return base


def get_location_rain_threshold_certainty_threshold(filename):
    # Use regular expressions to extract location, threshold, and certainty
    # Check for the 'ES_Base_prediction.out' pattern first
    if "Base_prediction" in filename:
        location = filename[0:2]
        threshold = None
        certainty = None
    else:
        # Use regular expressions to extract location, threshold, and certainty
        match = re.search(r"([A-Za-z]+)_(\d+(\.\d+)?)_mm_(\d+\.\d+)_cert", filename)

        if match:
            try:
                location = match.group(1)  # Location (e.g., 'ES')
                threshold = float(match.group(2))  # Threshold (e.g., 0.5)
                certainty = float(match.group(4))  # Certainty (e.g., 0.825)
            except TypeError:
                print(match)
        else:
            print(filename)
            location = None
            threshold = None
            certainty = None
    return location, threshold, certainty


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Total duration = {(end-start) / 60}")
