import datetime as dt
import pandas as pd
import numpy as np
import swmm_api as sa
import os as os
from storage import RZ_storage, ConcentrationStorage
import re
import time
from emprical_sewer_wq import EmpericalSewerWQ
from data.concentration_curves import concentration_dict_ES, concentration_dict_RZ

states = pd.read_csv(rf"simulation_states.csv", index_col=0, parse_dates=True)
filenames = os.listdir(rf"data/SWMM/outfile_saved")
path = os.path.join("data", "SWMM", "outfile_saved")


def main():
    results = {
        "location": [],
        "rain_threshold": [],
        "certainty_threshold": [],
        "OF_1_FD": [],
        "OF_1_outflow_2": [],
        "OF_2_cso": [],
        "OF_3_margin_05": [],
        "OF_3_margin_15": [],
        "OF_4_COD_part": [],
        "OF_4_COD_sol": [],
        "OF_4_X_TSS_sew": [],
        "OF_4_NH4_sew": [],
        "OF_4_PO4_sew": [],
    }
    for filename in filenames:
        location, threshold, certainty = (
            get_location_rain_threshold_certainty_threshold(filename)
        )
        results["location"].append(location)
        results["rain_threshold"].append(threshold)
        results["certainty_threshold"].append(certainty)

        output = sa.read_out_file(os.path.join(path, filename)).to_frame()
        break

        state = states.loc[:, find_collumn_name_from_filename(filename)]
        output, state = output.align(
            state, join="inner", axis=0
        )  # Align on the index (rows)

        outflow, FD, outflow_ideal = get_catchment_specific_flow_and_FD(
            output, location
        )
        OF_1_FD, OF_1_outflow_2 = objective_function_1(
            outflow, FD, state, outflow_ideal, output
        )
        results["OF_1_FD"].append(OF_1_FD)
        results["OF_1_outflow_2"].append(OF_1_outflow_2)

        OF_2_cso = objective_function_2(output, location)
        results["OF_2_cso"].append(OF_2_cso)

        OF_3_margin_05 = objective_function_3(outflow, outflow_ideal, margin=1.05)
        OF_3_margin_15 = objective_function_3(outflow, outflow_ideal, margin=1.5)
        results["OF_3_margin_05"].append(OF_3_margin_05)
        results["OF_3_margin_15"].append(OF_3_margin_15)

        OF_4_avg_load = objective_function_4(output, FD, location)
        results["OF_4_COD_part"].append(OF_4_avg_load["COD_part"])
        results["OF_4_COD_sol"].append(OF_4_avg_load["COD_sol"])
        results["OF_4_X_TSS_sew"].append(OF_4_avg_load["X_TSS_sew"])
        results["OF_4_NH4_sew"].append(OF_4_avg_load["NH4_sew"])
        results["OF_4_PO4_sew"].append(OF_4_avg_load["PO4_sew"])

        # One more for WWTP effluent

    # Convert the results dictionary to a pandas DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.round(3)
    # Save the DataFrame as a CSV file
    results_df.to_csv("objective_function_values.csv", index=False)


def objective_function_1(outflow, FD, state, outflow_ideal, output):
    wanted_state_outflow = np.logical_or(
        state.values == "dwf", state.values == "transition"
    )
    wanted_state_FD = get_first_wwf_state(state)

    initial_wwf_FD = FD[wanted_state_FD]

    mask = filted_high_inflows(output, outflow_ideal, outflow, wanted_state_outflow)
    filtered = outflow[mask]

    return np.mean(initial_wwf_FD), filtered.var()


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


def objective_function_4(output, FDs, location):
    if location == "ES":
        WQ_model = EmpericalSewerWQ(
            concentration_dict=concentration_dict_ES,
            COD_av=546,
            CODs_av=158,
            NH4_av=44,
            PO4_av=7.1,
            TSS_av=255,
            Q_95_av=70800,
            alpha_CODs=0.8,
            alpha_NH4=0.8,
            alpha_PO4=0.8,
            alpha_TSS=0.8,
            beta_CODs=0.8,
            beta_NH4=0.8,
            beta_PO4=0.8,
            beta_TSS=0.8,
        )

        inflow = output.node.pipe_ES.total_inflow * 3600 * 24
        timesteps = inflow.index
        outflow = output.link.P_eindhoven_out.flow * 3600 * 24
    else:
        WQ_model = EmpericalSewerWQ(
            concentration_dict=concentration_dict_RZ,
            COD_av=573,
            CODs_av=206,
            NH4_av=44,
            PO4_av=7.1,
            TSS_av=203,
            Q_95_av=58800,
            alpha_CODs=0.8,
            alpha_NH4=0.8,
            alpha_PO4=0.8,
            alpha_TSS=0.8,
            beta_CODs=0.8,
            beta_NH4=0.8,
            beta_PO4=0.8,
            beta_TSS=0.8,
            proc4_slope1_CODs=0.288,
            proc4_slope1_NH4=0.288,
            proc4_slope1_PO4=0.288,
            proc4_slope2_CODs=0.576,
            proc4_slope2_NH4=0.576,
            proc4_slope2_PO4=0.576,
            Q_proc6=120000,
            proc6_slope2_COD=864,
            proc6_slope2_TSS=864,
            proc6_t1_COD=3,
            proc6_t1_TSS=3,
            proc6_t2_COD=12,
            proc6_t2_TSS=12,
            proc7_slope1_COD=23040,
            proc7_slope1_TSS=23040,
            proc7_slope2_COD=5760,
            proc4_slope2_TSS=5760,
        )
        inflow = (
            (output.node["Nod_112"].total_inflow + output.node["Nod_104"].total_inflow)
            * 3600
            * 24
        )
        timesteps = inflow.index
        outflow = output.link.P_riool_zuid_out.flow * 3600 * 24
    concentration_storage = ConcentrationStorage()

    load_df = pd.DataFrame()
    for time, FD, Qin, Qout in zip(timesteps, FDs, inflow, outflow):
        WQ_model.update(time, Qin, FD)
        pollutant_flow = WQ_model.get_latest_log()

        concentration_storage.update_in(Qin, pollutant_flow)

        out_conc = concentration_storage.update_out(Qout, FD)

        row = pd.DataFrame([out_conc], index=[time])
        load_df = pd.concat([load_df, row])

    variance_dict = {}

    for pollutant in load_df.columns:
        series = load_df[pollutant]
        threshold = series.quantile(0.10)
        filtered = series[series >= threshold]
        # filtered.plot()
        variance_dict[pollutant] = filtered.var()

    return pd.Series(variance_dict, name="variance_below_95%")


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


def filted_high_inflows(output, outflow_ideal, outflow, wanted_state_outflow):
    inflow = output.node.pipe_ES.total_inflow
    threshold = 2 * outflow_ideal

    mask = pd.Series(True, index=outflow.index)
    filter_mode = False

    for i in range(len(outflow)):
        if filter_mode:
            mask.iloc[i] = False
            if wanted_state_outflow[i]:
                filter_mode = False
        elif inflow.iloc[i] > threshold:
            mask.iloc[i] = False
            filter_mode = True
    return mask


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


# if __name__ == "__main__":
# start = time.time()
# main()
# end = time.time()
# print(f"Total duration = {(end-start) / 60}")
