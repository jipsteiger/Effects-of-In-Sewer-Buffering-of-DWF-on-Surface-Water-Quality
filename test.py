import pyswmm as ps
import datetime as dt
import pandas as pd


def get_precipitation():
    return pd.read_csv(
        r"data\precipitation\csv_selected_area_euradclim\2024_5_min_precipitation_data.csv",
        index_col=0,
        parse_dates=True,
    )


def get_forecasts():
    df = pd.read_csv(
        r"data\precipitation\csv_forecasts\forecast_data.csv",
        usecols=[1, 2, 3, 4],
        index_col=1,
        parse_dates=True,
    )
    df.date_of_forecast = pd.to_datetime(df.date_of_forecast)
    return df


def set_simulation_settings(sim, step_size, report_start, start_time, end_time):
    sim.step_advance(step_size)
    sim.report_start = report_start
    sim.start_time = start_time
    sim.end_time = end_time


def get_links_and_nodes(sim):
    links = ps.Links(sim)
    nodes = ps.Nodes(sim)
    return links, nodes


def init_virtual_storage(nodes):
    virtual_storages = [node for node in nodes if "_vr_storage" in node.nodeid]
    virtual_storage_inflow = {node.nodeid: [] for node in virtual_storages}
    return virtual_storages, virtual_storage_inflow


def handle_virtual_storage(
    virtual_storages, virtual_storage_inflow, links, virtual_pump_max
):
    for virtual_storage in virtual_storages:
        nodeid = virtual_storage.nodeid
        virtual_storage_inflow[nodeid].append(virtual_storage.total_inflow)

        catchment_delay = int(nodeid.split("_")[-1])
        rounded_steps = round(catchment_delay / 5)

        if len(virtual_storage_inflow[nodeid]) == rounded_steps:
            pump_name = nodeid.replace("_vr_storage_", "_d_")
            delay_outflow = virtual_storage_inflow[nodeid].pop(0)
            links["P_" + pump_name].target_setting = delay_outflow / virtual_pump_max


def handle_c_119_flows(nodes, links, virtual_pump_max):
    J119_inflow = nodes["J119"].total_inflow
    bucket_inflows = ["P_c_119_fr_040", "P_c_119_fr_028", "P_c_119_fr_032"]

    for pump in bucket_inflows:
        fraction = float(pump.split("_")[-1]) / 100
        links[pump].target_setting = (fraction * J119_inflow) / virtual_pump_max


def run_simulation(
    model_path: str,
    step_size: int,
    report_start: dt.datetime,
    start_time: dt.datetime,
    end_time: dt.datetime,
    virtual_pump_max: int = 10,
):
    precipitation = get_precipitation()

    with ps.Simulation(model_path) as sim:
        set_simulation_settings(sim, step_size, report_start, start_time, end_time)
        links, nodes = get_links_and_nodes(sim)
        virtual_storages, virtual_storage_inflow = init_virtual_storage(nodes)

        for step in sim:
            print(sim.current_time)
            handle_virtual_storage(
                virtual_storages, virtual_storage_inflow, links, virtual_pump_max
            )
            handle_c_119_flows(nodes, links, virtual_pump_max)


MODEL_NAME = "model_jip"
run_simulation(
    model_path=rf"data\SWMM\{MODEL_NAME}.inp",
    step_size=300,
    report_start=dt.datetime(year=2024, month=7, day=1),
    start_time=dt.datetime(year=2024, month=7, day=1),
    end_time=dt.datetime(year=2024, month=7, day=3),
    virtual_pump_max=10,
)
