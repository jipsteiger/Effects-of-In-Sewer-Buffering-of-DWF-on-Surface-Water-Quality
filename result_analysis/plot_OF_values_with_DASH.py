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
    "OF_1_outflow": "Mean MSE of ideal outflow",
    "OF_1_FD": "Mean FD at start storm-event",
    "OF_2_cso": "Total CSO volume",
    "OF_3_margin_025": "Time of ideal outflow with 2.5 % margin",
    "OF_3_margin_05": "Time of ideal outflow with 5 % margin",
    "OF_3_margin_10": "Time of ideal outflow with 10 % margin",
    "OF_3_margin_15": "Time of ideal outflow with 15 % margin",
}

# Normalize and pivot
heatmaps = {"ES": {}, "RZ": {}}
for col in OF_columns:
    ES_df[f"{col}_norm"] = ES_df[col].values / ES_baseline[col].values
    RZ_df[f"{col}_norm"] = RZ_df[col].values / RZ_baseline[col].values

    heatmaps["ES"][col] = ES_df.pivot(
        index="rain_threshold", columns="certainty_threshold", values=f"{col}_norm"
    )
    heatmaps["RZ"][col] = RZ_df.pivot(
        index="rain_threshold", columns="certainty_threshold", values=f"{col}_norm"
    )

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
        dcc.Graph(id="heatmap-graph"),
        html.Div(
            id="click-output",
            style={"textAlign": "center", "marginTop": "20px", "fontSize": "18px"},
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
    fig = px.imshow(
        z_data.values,
        x=z_data.columns,
        y=z_data.index,
        labels=dict(x="Certainty Threshold", y="Rain Threshold", color="Normalized"),
        color_continuous_scale="Blues",
        text_auto=True,
        aspect="auto",  # Let it auto-scale
    )
    fig.update_layout(title=f"{catchment} - {titles.get(of_col, of_col)}", title_x=0.5)
    fig.update_traces(
        hovertemplate="Rain: %{y}<br>Certainty: %{x}<br>Normalized: %{z:.2f}"
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
