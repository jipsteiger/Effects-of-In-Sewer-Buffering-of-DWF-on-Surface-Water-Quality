import pandas as pd
import matplotlib.pyplot as plt

start_base = pd.Timestamp("2024-01-01")
time_windows = {
    "Dry": (pd.Timestamp("2024-08-08"), pd.Timestamp("2024-08-13")),
    "Wet": (pd.Timestamp("2024-07-02"), pd.Timestamp("2024-07-07")),
}


# Load function for each catchment
def load_scenario_data(catchment):
    def load_file(suffix, scenario_name):
        df = pd.read_csv(
            rf"output_swmm\{suffix}_out_{catchment}_{scenario_name}.csv",
            decimal=",",
            delimiter=";",
            index_col=0,
        )
        df["timestamp"] = start_base + pd.to_timedelta(df.index.astype(float), unit="D")
        df.set_index("timestamp", inplace=True)
        df.index = df.index.round("5min")
        return df

    return {
        "No RTC": load_file("07-03_15-41", "No_RTC"),
        "RTC": load_file("06-04_11-40", "RTC"),
        "RTC with ensembles": load_file("06-04_12-10", "Ensemble_RTC"),
    }


# Load both catchments
scenarios_ES = load_scenario_data("ES")
scenarios_RZ = load_scenario_data("RZ")

# Plot labels
labels = {
    "NH4_sew": "NH4 Load [kg/d]",
}
labels_conc = {
    "NH4_sew": "NH4 Concentration [mg/L]",
}
linestyles = {"ES": "-", "RZ": "--"}
colors = {
    "No RTC": "tab:blue",
    "RTC": "tab:green",
    "RTC with ensembles": "tab:orange",
}

# Loop over pollutants
for key in labels:
    include_conc = key != "H2O_sew"
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)

    for col_i, (period_label, (start, end)) in enumerate(time_windows.items()):
        for scenario_label in ["No RTC", "RTC", "RTC with ensembles"]:
            for catchment_label, scenario_dict in [
                ("ES", scenarios_ES),
                ("RZ", scenarios_RZ),
            ]:
                df = scenario_dict[scenario_label].loc[start:end]
                if key not in df.columns or "H2O_sew" not in df.columns:
                    continue

                # Plot load
                ax_load = axes[0][col_i]
                ax_load.plot(
                    df.index,
                    abs(df[key]) / 1000,  # g/d to mg/s
                    label=f"{scenario_label} - {catchment_label}",
                    color=colors[scenario_label],
                    linestyle=linestyles[catchment_label],
                )
                ax_load.set_title(f"{period_label} Period - Load")
                ax_load.set_ylabel(labels[key])
                ax_load.grid(True)

                # Plot concentration
                if include_conc:
                    ax_conc = axes[1][col_i]
                    ax_conc.plot(
                        df.index,
                        (df[key] / df["H2O_sew"]) * 1e6,
                        label=f"{scenario_label} - {catchment_label}",
                        color=colors[scenario_label],
                        linestyle=linestyles[catchment_label],
                    )
                    ax_conc.set_title(f"{period_label} Period - Concentration")
                    ax_conc.set_ylabel(labels_conc[key])
                    ax_conc.grid(True)

    # Final layout
    for ax in axes.flatten():
        ax.set_xlabel("Date")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=45)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
