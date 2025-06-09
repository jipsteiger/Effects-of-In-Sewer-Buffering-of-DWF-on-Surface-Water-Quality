import pandas as pd
import plotly.express as px
import dash
from dash import html, dcc, Output, Input
import dash_bootstrap_components as dbc
from result_analysis.plot_OF_values import plot_graphs_of_selection
import webbrowser
import threading

# Load Data
df = pd.read_csv("objective_function_values.csv")

# Filter ES and RZ
ES_df = df[df.location == "ES"]
RZ_df = df[df.location == "RZ"]
ES_baseline = ES_df[ES_df.isna().any(axis=1)]
ES_df = ES_df.dropna()
RZ_baseline = RZ_df[RZ_df.isna().any(axis=1)]
RZ_df = RZ_df.dropna()

# Objective Function (OF) columns
OF_columns = [col for col in ES_df.columns if "OF" in col]
titles = {
    "OF_1_outflow": "Var MSE of ideal outflow (lower is better)",
    "OF_1_FD": "Mean FD at start storm-event (lower is better)",
    "OF_1_outflow_2": "VAR of ideal outflow is inflow < 2x ideal and not WWF afterwards (lower is better)",  # Gets variance when
    "OF_2_cso": "Total CSO volume (lower is better)",
    "OF_3_margin_025": "Time of ideal outflow with 2.5 % margin (lower is better)",
    "OF_3_margin_05": "Time of ideal outflow with 5 % margin (lower is better)",
    "OF_3_margin_10": "Time of ideal outflow with 10 % margin (lower is better)",
    "OF_3_margin_15": "Time of ideal outflow with 15 % margin (lower is better)",
    "OF_4_COD_part": "Var COD part load to WWTP (lower is better)",
    "OF_4_COD_sol": "Var COD sol load to WWTP (lower is better)",
    "OF_4_X_TSS_sew": "Var X TSS sew load to WWTP (lower is better)",
    "OF_4_NH4_sew": "Var NH4 sew load to WWTP (lower is better)",
    "OF_4_PO4_sew": "Var PO4 sew load to WWTP (lower is better)",
    "score": "Mean score of selected components (higher is better)",
}

# Normalize and pivot
heatmaps = {"ES": {}, "RZ": {}}
for col in OF_columns:
    ES_df[f"{col}_norm"] = ES_df[col].values / ES_baseline[col].values
    RZ_df[f"{col}_norm"] = RZ_df[col].values / RZ_baseline[col].values

    if "margin" in col:
        ES_df[f"{col}_norm"] = 1 / ES_df[f"{col}_norm"]
        RZ_df[f"{col}_norm"] = 1 / RZ_df[f"{col}_norm"]

    heatmaps["ES"][col] = ES_df.pivot(
        index="rain_threshold", columns="certainty_threshold", values=f"{col}_norm"
    )
    heatmaps["RZ"][col] = RZ_df.pivot(
        index="rain_threshold", columns="certainty_threshold", values=f"{col}_norm"
    )


ES_df["load_mean_norm"] = ES_df.loc[
    :,
    [
        "OF_4_COD_part_norm",
        "OF_4_COD_sol_norm",
        "OF_4_X_TSS_sew_norm",
        "OF_4_NH4_sew_norm",
        "OF_4_PO4_sew_norm",
    ],
].mean(axis=1)
ES_df["OF_3_margin_025_norm_inv"] = 1 / ES_df["OF_3_margin_025_norm"]

RZ_df["load_mean_norm"] = RZ_df.loc[
    :,
    [
        "OF_4_COD_part_norm",
        "OF_4_COD_sol_norm",
        "OF_4_X_TSS_sew_norm",
        "OF_4_NH4_sew_norm",
        "OF_4_PO4_sew_norm",
    ],
].mean(axis=1)
RZ_df["OF_3_margin_025_norm_inv"] = 1 / RZ_df["OF_3_margin_025_norm"]

# Define weights (they must sum to 1, or normalize them after)
weights = {
    "OF_1_FD_norm": 0.1,
    "OF_1_outflow_2_norm": 0.15,
    "OF_2_cso_norm": 0.4,
    "load_mean_norm": 0.2,
    "OF_3_margin_025_norm_inv": 0.15,
}

# Make sure weights sum to 1
total_weight = sum(weights.values())
weights = {k: v / total_weight for k, v in weights.items()}

# Apply weighted average for ES
ES_df["score"] = sum(ES_df[col] * weight for col, weight in weights.items())

# Apply weighted average for RZ
RZ_df["score"] = sum(RZ_df[col] * weight for col, weight in weights.items())


heatmaps["ES"]["score"] = ES_df.pivot(
    index="rain_threshold", columns="certainty_threshold", values=f"score"
)
heatmaps["RZ"]["score"] = RZ_df.pivot(
    index="rain_threshold", columns="certainty_threshold", values=f"score"
)
OF_columns.append("score")

"""
GPT generated WebUI
"""

# Create app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    [
        html.H2(
            "Objective Function Heatmaps (Normalized to Baseline)",
            style={"textAlign": "center"},
        ),
        dcc.Graph(id="heatmap-graph"),
        html.Div(
            id="click-output",
            style={"textAlign": "center", "marginTop": "20px", "fontSize": "18px"},
        ),
        html.Div(
            [
                html.Label("Select Catchment:"),
                dcc.Dropdown(
                    id="catchment-select",
                    options=[
                        {"label": "ES", "value": "ES"},
                        {"label": "RZ", "value": "RZ"},
                    ],
                    value="ES",
                ),
                html.Label("Select Objective Function:"),
                dcc.Dropdown(
                    id="of-select",
                    options=[
                        {"label": titles.get(col, col), "value": col}
                        for col in OF_columns
                    ],
                    value=OF_columns[0],
                ),
            ],
            style={"width": "50%", "margin": "auto"},
        ),
    ]
)


@app.callback(
    Output("heatmap-graph", "figure"),
    Input("catchment-select", "value"),
    Input("of-select", "value"),
)
def update_heatmap(catchment, of_col):
    z_data = heatmaps[catchment][of_col]
    z_values = z_data.values
    min_val = z_values.min()

    # Find indices of minimum value(s)
    min_indices = [
        (i, j)
        for i in range(z_values.shape[0])
        for j in range(z_values.shape[1])
        if z_values[i, j] == min_val
    ]

    fig = px.imshow(
        z_values,
        x=z_data.columns,
        y=z_data.index,
        labels=dict(x="Confidence Level", y="Rain Threshold [mm]", color="Normalized"),
        color_continuous_scale="Blues",
        text_auto=".4f",
        aspect="auto",
    )

    # Overlay scatter to highlight min value
    for i, j in min_indices:
        fig.add_scatter(
            x=[z_data.columns[j] - 0.01],
            y=[z_data.index[i]],
            mode="markers",
            marker=dict(
                color="green",
                size=10,
                symbol="star",
                line=dict(width=0.5, color="black"),
            ),
            name="Min Value",
            showlegend=False,
            hoverinfo="skip",  # prevents tooltip duplication
        )

    fig.update_layout(title=f"{catchment} - {titles.get(of_col, of_col)}", title_x=0.5)
    fig.update_traces(
        selector=dict(type="heatmap"),
        hovertemplate="Rain: %{y}<br>Certainty: %{x}<br>Normalized: %{z:.2f}",
    )

    return fig


@app.callback(
    Output("click-output", "children"),
    Input("heatmap-graph", "clickData"),
    Input("catchment-select", "value"),
    Input("of-select", "value"),
)
def display_click_data(clickData, catchment, of_col):
    if clickData is None:
        return "Click on a heatmap cell to get details."

    point = clickData["points"][0]
    rain = point["y"]
    certainty = point["x"]
    value = point["z"]

    # Call a Python function
    your_function(catchment, of_col, rain, certainty, value)

    return f"Clicked: Catchment={catchment}, OF={of_col}, Rain={rain}, Certainty={certainty}, Normalized={value:.2f}"


def your_function(location, OF_name, rain_threshold, certainty_threshold, value):
    filename = f"{location}_{rain_threshold}_mm_{certainty_threshold}_cert"
    plot_graphs_of_selection([filename])


if __name__ == "__main__":
    plot_graphs_of_selection(["ES_Base_prediction"])

    # Open browser in a separate thread to avoid blocking
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:8050")).start()

    app.run(debug=True)
