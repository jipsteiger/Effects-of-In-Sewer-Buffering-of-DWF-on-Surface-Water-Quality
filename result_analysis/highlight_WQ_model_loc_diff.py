import numpy as np
import matplotlib.pyplot as plt

# --- Simulation settings ---
n_steps = 500
t = np.arange(n_steps)
omega = 2 * np.pi / 100  # cycle every 100 steps

# --- Generate inflow Q and concentration C (sinusoids with offset) ---
Q = 5 + 4 * np.sin(omega * t)  # Inflow
C = 2 + 1 * np.sin(omega * t - np.pi / 3)  # Concentration with phase shift

Q_mean = np.mean(Q)  # Constant outflow

# --- Tank buffer simulation ---
V = 0.0
L = 0.0
C_out_list = []

for q_in, c_in in zip(Q, C):
    load_in = q_in * c_in
    V += q_in
    L += load_in

    q_out = min(V, Q_mean)
    c_out = L / V if V > 0 else 0
    load_out = q_out * c_out

    V -= q_out
    L -= load_out

    C_out_list.append(c_out)

# --- Define range to plot: timesteps 400–500 ---
plot_start = 400
plot_end = n_steps
t_plot = t[plot_start:plot_end]
Q_plot = Q[plot_start:plot_end]
C_plot = C[plot_start:plot_end]
C_out_plot = C_out_list[plot_start:plot_end]

# --- Plot 1: Q and C on twin y-axes ---
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

ax1.plot(t_plot, Q_plot, "k", label="Diurnal flow")
ax2.plot(t_plot, C_plot, "k--", label="Diurnal concentration")

ax1.set_xlabel("Time")
ax1.set_ylabel("Flow")
ax2.set_ylabel("Concentration")

ax1.tick_params(axis="y")
ax2.tick_params(axis="y")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

fig1.tight_layout()

# --- Plot 2: Q_mean and both concentrations ---
fig2, ax3 = plt.subplots(figsize=(10, 5))
ax4 = ax3.twinx()

ax3.axhline(Q_mean, color="k", linestyle="-", label="Ideal flat outflow")
ax4.plot(t_plot, C_plot, "k--", linestyle="--", label="Unmixed concentration")
ax4.plot(t_plot, C_out_plot, "k.", linewidth=2, label="Mixed concentration")

ax3.set_xlabel("Time")
ax3.set_ylabel("Flow")
ax4.set_ylabel("Concentration")

ax3.tick_params(axis="y")
ax4.tick_params(axis="y")

# Combine legends
lines_1, labels_1 = ax3.get_legend_handles_labels()
lines_2, labels_2 = ax4.get_legend_handles_labels()
ax4.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

fig2.tight_layout()

plt.show()
