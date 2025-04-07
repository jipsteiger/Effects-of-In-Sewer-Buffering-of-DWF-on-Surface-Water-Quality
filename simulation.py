import pyswmm as ps
import datetime as dt
import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
)


class Simulation:
    def __init__(
        self,
        model_path: str,
        step_size: int,
        report_start: dt.datetime,
        start_time: dt.datetime,
        end_time: dt.datetime,
        virtual_pump_max: int = 10,
        ES_out_max: float = 3.888,
        ES_out_ideal: float = 0.663,
        RZ_out_max: float = 4.7222,
        RZ_out_ideal: float = 0.5218,
    ):
        self.model_path = model_path
        self.step_size = step_size
        self.report_start = report_start
        self.start_time = start_time
        self.end_time = end_time

        self.virtual_pump_max = virtual_pump_max

        self.precipitation_forecast = get_precipitation()
        self.current_forecast = None
        self.last_forecast_time = None

        self.ES_out_max = ES_out_max
        self.ES_out_ideal = ES_out_ideal
        self.RZ_out_max = RZ_out_max
        self.RZ_out_ideal = RZ_out_ideal

        self.setting_time = []
        self.ES_setting = []
        self.RZ_setting = []

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
            self.real_time_control()
            self.track_control_settings()

        self.save_settings()

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

    def track_control_settings(self):
        self.ES_setting.append(self.links["P_eindhoven_out"].target_setting)
        self.RZ_setting.append(self.links["P_riool_zuid_out"].target_setting)
        self.setting_time.append(self.sim.current_time)

    def real_time_control(self):
        self.get_forecast()

        ES_rain_forecast, RZ_rain_forecast = self.rain_predicted()

        if not ES_rain_forecast:
            self.links["P_eindhoven_out"].target_setting = (
                self.ES_out_ideal / self.ES_out_max
            )
        else:
            self.links["P_eindhoven_out"].target_setting = 1

        if not RZ_rain_forecast:
            self.links["P_riool_zuid_out"].target_setting = (
                self.RZ_out_ideal / self.RZ_out_max
            )
        else:
            self.links["P_riool_zuid_out"].target_setting = 1

    def get_forecast(self):
        time = self.sim.current_time.replace(minute=0, second=0, microsecond=0)

        if time.hour % 6 == 0 and self.last_forecast_time != time:
            end_time = time + dt.timedelta(hours=48)
            self.current_forecast = (
                self.precipitation_forecast.loc[time:end_time].resample("h").sum()
            )
            self.last_forecast_time = time

    def rain_predicted(self):
        time = self.sim.current_time.replace(minute=0, second=0, microsecond=0)
        end_time = time + dt.timedelta(hours=2)

        ES_forecast = self.current_forecast.loc[time:end_time, "ES"].sum()
        RZ_forecast = (
            self.current_forecast.loc[time:end_time, ["GE", "RZ1", "RZ2"]].sum().sum()
        )
        return ES_forecast > 0, RZ_forecast > 0

    def save_settings(self):
        df = pd.DataFrame(
            {
                "ES_setting CMS": np.array(self.ES_setting) * self.ES_out_max,
                "RZ_setting CMS": np.array(self.RZ_setting) * self.RZ_out_max,
            },
            index=self.setting_time,
        )

        df.to_csv("swmm_output/target_setting.csv")


def get_precipitation():
    return pd.read_csv(
        rf"data\precipitation\csv_selected_area_euradclim\2024_5_min_precipitation_data.csv",
        index_col=0,
        parse_dates=True,
    )


def get_forecasts():
    df = pd.read_csv(
        rf"data\precipitation\csv_forecasts\forecast_data.csv",
        usecols=[1, 2, 3, 4],
        index_col=1,
        parse_dates=True,
    )
    df.date_of_forecast = pd.to_datetime(df.date_of_forecast)
    return df


if __name__ == "__main__":
    MODEL_NAME = "model_jip"
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
        save=False, plot_rain=True, target_setting=True, storage=True
    )
