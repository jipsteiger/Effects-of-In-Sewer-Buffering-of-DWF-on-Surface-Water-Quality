import pyswmm as ps
import datetime as dt
import pandas as pd


class Simulation:
    def __init__(
        self,
        model_path: str,
        step_size: int,
        report_start: dt.datetime,
        start_time: dt.datetime,
        end_time: dt.datetime,
        virtual_pump_max: int = 10,
    ):
        self.model_path = model_path
        self.step_size = step_size
        self.report_start = report_start
        self.start_time = start_time
        self.end_time = end_time

        self.virtual_pump_max = virtual_pump_max

        self.precipitation_forecast = get_precipitation()

    def start_simulation(self):
        with ps.Simulation(
            self.model_path,
        ) as self.sim:
            self.set_simulation_settings()
            self.get_links_and_nodes()
            self.init_virtual_storage()

            self.simulation_steps()

    def set_simulation_settings(self):
        self.sim.step_advance(self.step_size)
        self.sim.report_start = self.report_start
        self.sim.start_time = self.start_time
        self.sim.end_time = self.end_time

    def get_links_and_nodes(self):
        self.links = ps.Links(self.sim)
        self.nodes = ps.Nodes(self.sim)

    def init_virtual_storage(self):
        self.virtual_storages = [
            node for node in self.nodes if "_vr_storage" in node.nodeid
        ]
        self.virtual_storage_inflow = {}
        for virtual_storage in self.virtual_storages:
            self.virtual_storage_inflow[virtual_storage.nodeid] = []

    def simulation_steps(self):
        for step in self.sim:
            self.handle_virtual_storage()
            self.handle_c_119_flows()
            self.handle_geldrop_out_flows()

    def handle_virtual_storage(self):
        for virtual_storage in self.virtual_storages:
            self.virtual_storage_inflow[virtual_storage.nodeid].append(
                virtual_storage.total_inflow
            )

            catchment_delay = int(virtual_storage.nodeid.split("_")[-1])
            rounded_steps = round(catchment_delay / 5)
            if (
                len(self.virtual_storage_inflow[virtual_storage.nodeid])
                == rounded_steps
            ):
                pump_name = virtual_storage.nodeid.replace("_vr_storage_", "_d_")

                delay_outflow = self.virtual_storage_inflow[virtual_storage.nodeid].pop(
                    0
                )
                self.links["P_" + pump_name].target_setting = (
                    delay_outflow / self.virtual_pump_max
                )

    def handle_c_119_flows(self):
        J119_inflow = self.nodes["J119"].total_inflow

        bucket_inflows = ["P_c_119_fr_040", "P_c_119_fr_028", "P_c_119_fr_032"]

        for pump in bucket_inflows:
            fraction = (
                float(pump.split("_")[-1]) / 100
            )  # Get the inflow fraction from the pump name
            self.links[pump].target_setting = (
                fraction * J119_inflow
            ) / self.virtual_pump_max

    def handle_geldrop_out_flows(self):
        J1_inflow = self.nodes["J1"].total_inflow

        pumps = ["P_geldrop_split_1", "P_geldrop_split_2"]

        for pump in pumps:
            fraction = 0.5
            self.links[pump].target_setting = (
                fraction * J1_inflow
            ) / self.virtual_pump_max


def get_precipitation():
    return pd.read_csv(
        rf"data\precipitation\csv_selected_area_euradclim\2024_5_min_precipitation_data.csv",
        index_col=0,
        parse_dates=True,
    )


##################################################################################################
if __name__ == "__main__":
    MODEL_NAME = "model_jip"
    suffix = "Reg. sim"
    simulation = Simulation(
        model_path=rf"data\SWMM\{MODEL_NAME}.inp",
        step_size=300,
        report_start=dt.datetime(year=2024, month=7, day=1),
        start_time=dt.datetime(year=2024, month=7, day=1),
        end_time=dt.datetime(year=2024, month=7, day=10),
        virtual_pump_max=10,
    )
    simulation.start_simulation()

    from postprocess import PostProcess

    postprocess = PostProcess(model_name=MODEL_NAME)
    postprocess.plot_pumps(
        save=False, plot_rain=True, target_setting=True, suffix=suffix, storage=True
    )
