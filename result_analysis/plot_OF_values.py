import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import swmm_api as sa


def plot_graphs_of_selection(names=[]):
    from postprocess import PostProcess

    if len(names) == 0:
        names = ["ES_Base_prediction"]
        values = [[2.75, 0.9], [1, 0.75]]
        for value in values:
            names.append(f"ES_{value[0]}_mm_{value[1]}_cert")

    for name in names:
        suffix = name
        postprocess = PostProcess(model_name=name, path=rf"data\swmm\outfile_saved")
        postprocess.plot_pumps(
            save=False,
            plot_rain=True,
            target_setting=True,
            suffix=suffix,
            storage=True,
            title=name,
        )


def analyse_parameter_results_OLD():
    # Load data
    df = pd.read_csv("forecast_sensitivity_analysis_2.csv")

    # Filter ES DataFrame
    ES_df = df[df["ES_threshold"].notna()][
        [
            "ES_threshold",
            "certainty_threshold",
            "ES_flow",
            "ES_FD",
            "ES_cso",
            "ES_flow_sum",
            "ES_FD_sum",
            "ES_flow_always",
            "ES_flow_always_sum",
        ]
    ]

    # Filter RZ DataFrame
    RZ_df = df[df["RZ_threshold"].notna()][
        [
            "RZ_threshold",
            "certainty_threshold",
            "RZ_flow",
            "RZ_FD",
            "RZ_cso",
            "RZ_flow_sum",
            "RZ_FD_sum",
            "RZ_flow_always",
            "RZ_flow_always_sum",
        ]
    ]

    # Baseline values (regular results)
    ES_baseline = {
        "ES_flow": 0.049,
        "ES_FD": 2.215,
        "ES_flow_sum": 354.867,
        "ES_FD_sum": 15.507,
        "ES_flow_always": 1.144,
        "ES_flow_always_sum": 9883.462,
        "ES_cso": 30.332,
    }

    RZ_baseline = {
        "RZ_flow": 0.036,
        "RZ_FD": 1.434,
        "RZ_flow_sum": 266.293,
        "RZ_FD_sum": 8.601,
        "RZ_flow_always": 1.506,
        "RZ_flow_always_sum": 13008.808,
        "RZ_cso": 309.557,
    }

    # Normalize ES metrics
    ES_df["ES_flow_norm"] = ES_df["ES_flow"] / ES_baseline["ES_flow"]
    ES_df["ES_FD_norm"] = ES_df["ES_FD"] / ES_baseline["ES_FD"]
    ES_df["ES_cso_norm"] = ES_df["ES_cso"] / ES_baseline["ES_cso"]
    ES_df["score"] = ES_df["ES_flow_norm"] + ES_df["ES_FD_norm"] + ES_df["ES_cso_norm"]

    # Normalize RZ metrics
    RZ_df["RZ_flow_norm"] = RZ_df["RZ_flow"] / RZ_baseline["RZ_flow"]
    RZ_df["RZ_FD_norm"] = RZ_df["RZ_FD"] / RZ_baseline["RZ_FD"]
    RZ_df["RZ_cso_norm"] = RZ_df["RZ_cso"] / RZ_baseline["RZ_cso"]
    RZ_df["score"] = RZ_df["RZ_flow_norm"] + RZ_df["RZ_FD_norm"] + RZ_df["RZ_cso_norm"]

    # Normalize additional ES metrics
    ES_df["ES_flow_sum_norm"] = ES_df["ES_flow_sum"] / ES_baseline["ES_flow_sum"]
    ES_df["ES_FD_sum_norm"] = ES_df["ES_FD_sum"] / ES_baseline["ES_FD_sum"]
    ES_df["ES_flow_always_norm"] = (
        ES_df["ES_flow_always"] / ES_baseline["ES_flow_always"]
    )
    ES_df["ES_flow_always_sum_norm"] = (
        ES_df["ES_flow_always_sum"] / ES_baseline["ES_flow_always_sum"]
    )

    # Normalize additional RZ metrics
    RZ_df["RZ_flow_sum_norm"] = RZ_df["RZ_flow_sum"] / RZ_baseline["RZ_flow_sum"]
    RZ_df["RZ_FD_sum_norm"] = RZ_df["RZ_FD_sum"] / RZ_baseline["RZ_FD_sum"]
    RZ_df["RZ_flow_always_norm"] = (
        RZ_df["RZ_flow_always"] / RZ_baseline["RZ_flow_always"]
    )
    RZ_df["RZ_flow_always_sum_norm"] = (
        RZ_df["RZ_flow_always_sum"] / RZ_baseline["RZ_flow_always_sum"]
    )

    # Create pivot tables for heatmaps
    heatmaps = {
        "ES": {
            "flow": ES_df.pivot(
                index="ES_threshold",
                columns="certainty_threshold",
                values="ES_flow_norm",
            ),
            "FD": ES_df.pivot(
                index="ES_threshold", columns="certainty_threshold", values="ES_FD_norm"
            ),
            "CSO": ES_df.pivot(
                index="ES_threshold",
                columns="certainty_threshold",
                values="ES_cso_norm",
            ),
            "score": ES_df.pivot(
                index="ES_threshold", columns="certainty_threshold", values="score"
            ),
        },
        "RZ": {
            "flow": RZ_df.pivot(
                index="RZ_threshold",
                columns="certainty_threshold",
                values="RZ_flow_norm",
            ),
            "FD": RZ_df.pivot(
                index="RZ_threshold", columns="certainty_threshold", values="RZ_FD_norm"
            ),
            "CSO": RZ_df.pivot(
                index="RZ_threshold",
                columns="certainty_threshold",
                values="RZ_cso_norm",
            ),
            "score": RZ_df.pivot(
                index="RZ_threshold", columns="certainty_threshold", values="score"
            ),
        },
    }
    # Extend heatmaps
    heatmaps["ES"].update(
        {
            "flow_sum": ES_df.pivot(
                index="ES_threshold",
                columns="certainty_threshold",
                values="ES_flow_sum_norm",
            ),
            "FD_sum": ES_df.pivot(
                index="ES_threshold",
                columns="certainty_threshold",
                values="ES_FD_sum_norm",
            ),
            "flow_always": ES_df.pivot(
                index="ES_threshold",
                columns="certainty_threshold",
                values="ES_flow_always_norm",
            ),
            "flow_always_sum": ES_df.pivot(
                index="ES_threshold",
                columns="certainty_threshold",
                values="ES_flow_always_sum_norm",
            ),
        }
    )

    heatmaps["RZ"].update(
        {
            "flow_sum": RZ_df.pivot(
                index="RZ_threshold",
                columns="certainty_threshold",
                values="RZ_flow_sum_norm",
            ),
            "FD_sum": RZ_df.pivot(
                index="RZ_threshold",
                columns="certainty_threshold",
                values="RZ_FD_sum_norm",
            ),
            "flow_always": RZ_df.pivot(
                index="RZ_threshold",
                columns="certainty_threshold",
                values="RZ_flow_always_norm",
            ),
            "flow_always_sum": RZ_df.pivot(
                index="RZ_threshold",
                columns="certainty_threshold",
                values="RZ_flow_always_sum_norm",
            ),
        }
    )

    # Plot
    fig, axs = plt.subplots(2, 8, figsize=(60, 15))  # Wider for more plots
    titles = [
        "Normalized mean MSE of ideal outflow",
        "Normalized mean FD at start storm-event",
        "Normalized total CSO volume",
        "Combined Score",
        "Normalized sum MSE of ideal outflow",
        "Normalized sum FD at start storm-event",
        "Normalized mean always MSE of ideal outflow",
        "Normalized sum always MSE of ideal outflow",
    ]
    cmap = ["Blues", "Greens", "Oranges", "Purples", "PuBu", "BuGn", "YlGnBu", "YlOrBr"]
    keys = [
        "flow",
        "FD",
        "CSO",
        "score",
        "flow_sum",
        "FD_sum",
        "flow_always",
        "flow_always_sum",
    ]

    for i, key in enumerate(keys):
        # ES row
        sns.heatmap(
            heatmaps["ES"][key],
            cmap=cmap[i],
            annot=True,
            ax=axs[0, i],
            cbar_kws={"label": "Relative to Baseline"},
        )
        axs[0, i].set_title(f"ES {titles[i]}")
        axs[0, i].set_xlabel("Precipitation certainty threshold")
        axs[0, i].set_ylabel("ES catchment precipitation threshold [mm]")

        # RZ row
        sns.heatmap(
            heatmaps["RZ"][key],
            cmap=cmap[i],
            annot=True,
            ax=axs[1, i],
            cbar_kws={"label": "Relative to RTC with ideal forecast"},
        )
        axs[1, i].set_title(f"RZ {titles[i]}")
        axs[1, i].set_xlabel("Precipitation certainty threshold")
        axs[1, i].set_ylabel("RZ catchment precipitation threshold [mm]")

    fig.suptitle(
        "Metrics normalized to ideal RTC values:\n"
        "Includes flow, FD, CSO, and additional aggregate/RTC metrics.\n"
        "Values <1 indicate improved performance under non-ideal forecast conditions.",
        fontsize=16,
        y=1.05,
    )
    plt.tight_layout()
    plt.show()


def analyse_objective_functions():
    # Load data
    df = pd.read_csv("objective_function_values.csv")

    # Filter ES DataFrame
    ES_df = df[df.location == "ES"]
    # Filter RZ DataFrame
    RZ_df = df[df.location == "RZ"]

    # Baseline values (regular results)
    ES_baseline = ES_df[ES_df.isna().any(axis=1)]
    ES_df = ES_df.dropna()

    RZ_baseline = RZ_df[RZ_df.isna().any(axis=1)]
    RZ_df = RZ_df.dropna()

    # Define the OF column keys to loop over (add more keys if needed)
    OF_columns = [key for key in list(ES_df.keys()) if "OF" in key]
    titles = {
        "OF_1_outflow": "Mean MSE of ideal outflow",
        "OF_1_FD": "Mean FD at start storm-event",
        "OF_2_cso": "Total CSO volume",
        "OF_3_margin_025": "Time of ideal outflow with 2.5 % margin",
        "OF_3_margin_05": "Time of ideal outflow with 5 % margin",
        "OF_3_margin_10": "Time of ideal outflow with 10 % margin",
        "OF_3_margin_15": "Time of ideal outflow with 15 % margin",
    }

    # Create an empty dictionary for heatmaps
    heatmaps = {"ES": {}, "RZ": {}}

    # Normalize ES metrics and RZ metrics for each OF column
    for col in OF_columns:
        # Normalize ES metrics based on the OF column
        ES_df[f"{col}_norm"] = ES_df[col].values / ES_baseline[col].values

        # Normalize RZ metrics based on the OF column
        RZ_df[f"{col}_norm"] = RZ_df[col].values / RZ_baseline[col].values

        # Create pivot tables for each OF column for heatmaps
        heatmaps["ES"][col] = ES_df.pivot(
            index="rain_threshold", columns="certainty_threshold", values=f"{col}_norm"
        )
        heatmaps["RZ"][col] = RZ_df.pivot(
            index="rain_threshold", columns="certainty_threshold", values=f"{col}_norm"
        )

    # Plot
    fig, axs = plt.subplots(
        2, len(OF_columns), figsize=(60, 15)
    )  # Adjusted for OF columns + score
    cmap = [
        "Blues",
        "Greens",
        "Oranges",
        "Purples",
        "PuBu",
        "BuGn",
        "YlGnBu",
        "YlOrBr",
    ]  # Adjust as needed for each plot

    # Ensure cmap has the right length or assign default if not
    try:
        if len(cmap) < len(OF_columns):
            raise ValueError("cmap length does not match number of OF columns.")
    except Exception as e:
        print(f"Error in cmap setup: {e}")
        # Default cmap in case of mismatch
        cmap = ["Blues"] * len(OF_columns)

    # Ensure titles are provided for each OF column
    try:
        for col in OF_columns:
            if col not in titles:
                titles[col] = f"Normalized value for {col}"  # Default title if missing
    except Exception as e:
        print(f"Error in titles setup: {e}")
        # Default titles in case of error
        titles = {col: f"Normalized value for {col}" for col in OF_columns}

    # Plot the heatmaps for each OF column and the score
    for i, col in enumerate(OF_columns):
        # ES row
        sns.heatmap(
            heatmaps["ES"][col],
            cmap=cmap[i],
            annot=True,
            ax=axs[0, i],
            cbar_kws={"label": "Relative to Baseline"},
        )
        axs[0, i].set_title(f"ES {titles.get(col, col)}")
        axs[0, i].set_xlabel("Precipitation certainty threshold")
        axs[0, i].set_ylabel("ES catchment precipitation threshold [mm]")

        # RZ row
        sns.heatmap(
            heatmaps["RZ"][col],
            cmap=cmap[i],
            annot=True,
            ax=axs[1, i],
            cbar_kws={"label": "Relative to Baseline"},
        )
        axs[1, i].set_title(f"RZ {titles.get(col, col)}")
        axs[1, i].set_xlabel("Precipitation certainty threshold")
        axs[1, i].set_ylabel("RZ catchment precipitation threshold [mm]")

    fig.suptitle(
        "All metrics are normalized with regard to RTC values with perfect prediction (the baseline)",
        fontsize=16,
        y=1.05,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyse_objective_functions()
