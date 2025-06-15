from data.concentration_curves import (
    NH4_conc_ES,
    Q_95_norm_ES_adjusted,
    NH4_conc_RZ,
    Q_95_norm_RZ_adjusted,
)
from storage import ConcentrationStorage
import numpy as np
import pandas as pd
from datetime import timedelta

# --- 1. SETUP & PARAMETER CALCULATION ---

# Collect the 24-hour profile data first
hourly_inflow_rates = []
hourly_nh4_concs = []
for i in range(24):
    hour_key = f"H_{i}"
    hourly_inflow_rates.append(getattr(Q_95_norm_ES_adjusted, hour_key))
    # Assuming NH4_conc_ES is in g/m3. The original comment "g/d" is likely a typo for concentration.
    hourly_nh4_concs.append(getattr(NH4_conc_ES, hour_key))

# Convert to numpy arrays for vectorized calculations
hourly_inflow_rates = np.array(hourly_inflow_rates) * 0.663  # units: m3/s
hourly_nh4_concs = np.array(hourly_nh4_concs) * 44  # units: g/m3

# Calculate the true average inflow rate and average incoming load
avg_inflow_rate = np.mean(hourly_inflow_rates)  # m3/s
hourly_loads = hourly_inflow_rates * hourly_nh4_concs  # (m3/s) * (g/m3) = g/s
avg_incoming_load = np.mean(hourly_loads)  # g/s

print(f"Average Inflow Rate: {avg_inflow_rate:.4f} m3/s")
print(f"Average Incoming NH4 Load: {avg_incoming_load:.4f} g/s")

# --- 2. CONTROL PARAMETERS for the Storage Tank ---

# Our target is to have a constant outgoing load equal to the average incoming load.
target_outflow_load_NH4 = avg_incoming_load  # g/s

# Define the storage tank's operational parameters
V_target = (
    10000  # m3. The desired average volume in the tank. Choose a reasonable size.
)
Kp = 0.0001  # Proportional gain. A small value to start. Units are (m3/s) / m3 -> 1/s.

# --- 3. SIMULATION SETUP ---

# Total number of 5-minute steps in 7 days
num_steps = int((7 * 24 * 60) / 5)  # 2016 steps
timestep_seconds = 300  # 5 minutes in seconds

concentrationStorage = ConcentrationStorage()

# Dataframe to store results
outflow_list = []
V_list = []
load_out_list = []  # To check if our load is constant
concentration_df = pd.DataFrame(columns=["COD", "CODs", "TSS", "NH4", "PO4"])

# --- 4. SIMULATION LOOP ---

for step in range(num_steps):
    # Determine hour of day to index the 24-hour curves
    hour_of_day = (step * 5) // 60 % 24

    # Get inflow and concentration for the current time step
    flow_in = hourly_inflow_rates[hour_of_day]  # m3/s
    NH4_in_conc = hourly_nh4_concs[hour_of_day]  # g/m3

    inflow_concentrations = {
        "COD_part": 0,
        "COD_sol": 0,
        "X_TSS_sew": 0,
        "NH4_sew": NH4_in_conc,  # g/m3
        "PO4_sew": 0,
    }
    concentrationStorage.update_in(flow_in, inflow_concentrations, timestep_seconds)

    # Get the current state of the tank AFTER adding the inflow
    V, conc_state = (
        concentrationStorage.get_current_state()
    )  # V in m3, conc_state in g/m3
    V_list.append(V)

    # --- THIS IS THE CORRECTED CONTROL LOGIC ---

    current_NH4_conc = conc_state["NH4_sew"]

    # Avoid division by zero if tank is clean
    if current_NH4_conc > 0.1:
        # 1. Calculate the ideal outflow to meet the load target
        Q_for_load_target = (
            target_outflow_load_NH4 / current_NH4_conc
        )  # (g/s) / (g/m3) = m3/s
    else:
        Q_for_load_target = (
            avg_inflow_rate  # Fallback to average flow if concentration is zero
        )

    # 2. Calculate the volume correction flow (Proportional Controller)
    volume_error = V - V_target  # m3
    Q_volume_correction = Kp * volume_error  # (1/s) * m3 = m3/s

    # 3. Combine them to get the final outflow
    outflow = Q_for_load_target + Q_volume_correction
    # outflow = avg_inflow_rate

    # 4. Add a safety check: outflow cannot be negative
    outflow = max(0, outflow)

    # --- END OF CORRECTED LOGIC ---

    # Update storage with the calculated outflow
    # Assume update_out returns a dict of pollutant loads in g/s (if timestep_seconds is used internally for conversion from g/d)
    # or g/d (as per original comment). Let's calculate the load ourselves for clarity.
    output_loads_per_day = concentrationStorage.update_out(
        outflow, 0, timestep_seconds
    )  # Original call

    # For plotting, let's calculate the outgoing load in g/s to verify our controller
    actual_outgoing_load = outflow * current_NH4_conc  # (m3/s) * (g/m3) = g/s
    load_out_list.append(actual_outgoing_load)

    # Store results
    timestamp = pd.Timestamp("2025-01-01") + timedelta(minutes=5 * step)
    row_df = pd.DataFrame([output_loads_per_day], index=[timestamp])
    concentration_df = pd.concat([concentration_df, row_df])
    outflow_list.append(outflow)


import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Ensure the plot opens in the default browser
pio.renderers.default = "browser"

# Combine volume and outflow with concentration_df
concentration_df["Outflow"] = outflow_list
concentration_df["Volume"] = V_list

# Create subplots: 3 rows, shared x-axis
fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    subplot_titles=("NH4 Load", "Outflow", "NH4 Concentration", "Storage Volume"),
    vertical_spacing=0.1,
)

# --- Row 1: NH4 Concentration ---
fig.add_trace(
    go.Scatter(
        x=concentration_df.index,
        y=concentration_df["NH4_sew"],
        mode="lines",
        name="NH4 Concentration",
        line=dict(color="blue"),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=concentration_df.index,
        y=concentration_df["NH4_sew"] / concentration_df["Outflow"],
        mode="lines",
        name="NH4 Load",
        line=dict(color="red"),
    ),
    row=3,
    col=1,
)


# --- Row 2: Outflow ---
fig.add_trace(
    go.Scatter(
        x=concentration_df.index,
        y=concentration_df["Outflow"],
        mode="lines",
        name="Outflow",
        line=dict(color="green"),
    ),
    row=2,
    col=1,
)

# --- Row 3: Storage Volume ---
fig.add_trace(
    go.Scatter(
        x=concentration_df.index,
        y=concentration_df["Volume"],
        mode="lines",
        name="Volume",
        line=dict(color="orange"),
    ),
    row=4,
    col=1,
)

# Layout settings
fig.update_layout(
    height=900,
    title_text="NH4 Concentration, Outflow, and Storage Volume Over Time",
    xaxis_title="Time",
    template="plotly_white",
)

# Show plot in default web browser
fig.show()


#####################################################################################################

from data.concentration_curves import NH4_conc_ES, Q_95_norm_ES_adjusted
from storage import ConcentrationStorage
import numpy as np
import pandas as pd
from datetime import timedelta

# --- 1. SETUP & PARAMETER CALCULATION ---

# Collect the 24-hour profile data first
hourly_inflow_rates = []
hourly_nh4_concs = []
for i in range(24):
    hour_key = f"H_{i}"
    hourly_inflow_rates.append(getattr(Q_95_norm_RZ_adjusted, hour_key))
    # Assuming NH4_conc_ES is in g/m3. The original comment "g/d" is likely a typo for concentration.
    hourly_nh4_concs.append(getattr(NH4_conc_RZ, hour_key))

# Convert to numpy arrays for vectorized calculations
hourly_inflow_rates = np.array(hourly_inflow_rates) * 0.5218  # units: m3/s
hourly_nh4_concs = np.array(hourly_nh4_concs) * 44  # units: g/m3

# Calculate the true average inflow rate and average incoming load
avg_inflow_rate = np.mean(hourly_inflow_rates)  # m3/s
hourly_loads = hourly_inflow_rates * hourly_nh4_concs  # (m3/s) * (g/m3) = g/s
avg_incoming_load = np.mean(hourly_loads)  # g/s

print(f"Average Inflow Rate: {avg_inflow_rate:.4f} m3/s")
print(f"Average Incoming NH4 Load: {avg_incoming_load:.4f} g/s")

# --- 2. CONTROL PARAMETERS for the Storage Tank ---

# Our target is to have a constant outgoing load equal to the average incoming load.
target_outflow_load_NH4 = avg_incoming_load  # g/s

# Define the storage tank's operational parameters
V_target = 5000  # m3. The desired average volume in the tank. Choose a reasonable size.
Kp = 0.0001  # Proportional gain. A small value to start. Units are (m3/s) / m3 -> 1/s.

# --- 3. SIMULATION SETUP ---

# Total number of 5-minute steps in 7 days
num_steps = int((7 * 24 * 60) / 5)  # 2016 steps
timestep_seconds = 300  # 5 minutes in seconds

# Initialize the storage with the target volume and average concentration
# This prevents wild fluctuations at the start of the simulation.
avg_incoming_conc = avg_incoming_load / avg_inflow_rate  # g/m3
initial_concentrations = {
    "COD_part": 0,
    "COD_sol": 0,
    "X_TSS_sew": 0,
    "NH4_sew": avg_incoming_conc,
    "PO4_sew": 0,
}
concentrationStorage = ConcentrationStorage()

# Dataframe to store results
outflow_list = []
V_list = []
load_out_list = []  # To check if our load is constant
concentration_df = pd.DataFrame(columns=["COD", "CODs", "TSS", "NH4", "PO4"])

# --- 4. SIMULATION LOOP ---

for step in range(num_steps):
    # Determine hour of day to index the 24-hour curves
    hour_of_day = (step * 5) // 60 % 24

    # Get inflow and concentration for the current time step
    flow_in = hourly_inflow_rates[hour_of_day]  # m3/s
    NH4_in_conc = hourly_nh4_concs[hour_of_day]  # g/m3

    inflow_concentrations = {
        "COD_part": 0,
        "COD_sol": 0,
        "X_TSS_sew": 0,
        "NH4_sew": NH4_in_conc,  # g/m3
        "PO4_sew": 0,
    }
    concentrationStorage.update_in(flow_in, inflow_concentrations, timestep_seconds)

    # Get the current state of the tank AFTER adding the inflow
    V, conc_state = (
        concentrationStorage.get_current_state()
    )  # V in m3, conc_state in g/m3
    V_list.append(V)

    # --- THIS IS THE CORRECTED CONTROL LOGIC ---

    current_NH4_conc = conc_state["NH4_sew"]

    # Avoid division by zero if tank is clean
    if current_NH4_conc > 0.1:
        # 1. Calculate the ideal outflow to meet the load target
        Q_for_load_target = (
            target_outflow_load_NH4 / current_NH4_conc
        )  # (g/s) / (g/m3) = m3/s
    else:
        Q_for_load_target = (
            avg_inflow_rate  # Fallback to average flow if concentration is zero
        )

    # 2. Calculate the volume correction flow (Proportional Controller)
    volume_error = V - V_target  # m3
    Q_volume_correction = Kp * volume_error  # (1/s) * m3 = m3/s

    # 3. Combine them to get the final outflow
    outflow = Q_for_load_target + Q_volume_correction
    # outflow = avg_inflow_rate

    # 4. Add a safety check: outflow cannot be negative
    outflow = max(0, outflow)

    # --- END OF CORRECTED LOGIC ---

    # Update storage with the calculated outflow
    # Assume update_out returns a dict of pollutant loads in g/s (if timestep_seconds is used internally for conversion from g/d)
    # or g/d (as per original comment). Let's calculate the load ourselves for clarity.
    output_loads_per_day = concentrationStorage.update_out(
        outflow, 0, timestep_seconds
    )  # Original call

    # For plotting, let's calculate the outgoing load in g/s to verify our controller
    actual_outgoing_load = outflow * current_NH4_conc  # (m3/s) * (g/m3) = g/s
    load_out_list.append(actual_outgoing_load)

    # Store results
    timestamp = pd.Timestamp("2025-01-01") + timedelta(minutes=5 * step)
    row_df = pd.DataFrame([output_loads_per_day], index=[timestamp])
    concentration_df = pd.concat([concentration_df, row_df])
    outflow_list.append(outflow)


import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Ensure the plot opens in the default browser
pio.renderers.default = "browser"

# Combine volume and outflow with concentration_df
concentration_df["Outflow"] = outflow_list
concentration_df["Volume"] = V_list

# Create subplots: 3 rows, shared x-axis
fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    subplot_titles=("NH4 Load", "Outflow", "NH4 Concentration", "Storage Volume"),
    vertical_spacing=0.1,
)

# --- Row 1: NH4 Concentration ---
fig.add_trace(
    go.Scatter(
        x=concentration_df.index,
        y=concentration_df["NH4_sew"],
        mode="lines",
        name="NH4 Concentration",
        line=dict(color="blue"),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=concentration_df.index,
        y=concentration_df["NH4_sew"] / concentration_df["Outflow"],
        mode="lines",
        name="NH4 Load",
        line=dict(color="red"),
    ),
    row=3,
    col=1,
)


# --- Row 2: Outflow ---
fig.add_trace(
    go.Scatter(
        x=concentration_df.index,
        y=concentration_df["Outflow"],
        mode="lines",
        name="Outflow",
        line=dict(color="green"),
    ),
    row=2,
    col=1,
)

# --- Row 3: Storage Volume ---
fig.add_trace(
    go.Scatter(
        x=concentration_df.index,
        y=concentration_df["Volume"],
        mode="lines",
        name="Volume",
        line=dict(color="orange"),
    ),
    row=4,
    col=1,
)

# Layout settings
fig.update_layout(
    height=900,
    title_text="NH4 Concentration, Outflow, and Storage Volume Over Time",
    xaxis_title="Time",
    template="plotly_white",
)

# Show plot in default web browser
fig.show()
