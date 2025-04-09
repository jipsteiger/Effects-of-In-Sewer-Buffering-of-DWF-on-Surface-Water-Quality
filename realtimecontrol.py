import pyswmm as ps
import datetime as dt
import pandas as pd
import numpy as np
import logging
from pump_curves import EsPumpCurve, RzPumpCurve
from simulation import Simulation
from storage import Storage, RZ_storage

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
)


class RealTimeControl(Simulation):
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
        ES_ramp_steps: int = 36,
    ):
        super().__init__(
            model_path,
            step_size,
            report_start,
            start_time,
            end_time,
            virtual_pump_max,
        )
        self.current_forecast = None
        self.last_forecast_time = None

        self.ES_storage = Storage(165000)
        self.RZ_storage = RZ_storage()

        self.ES_out_max = ES_out_max
        self.ES_out_ideal = ES_out_ideal
        self.RZ_out_max = RZ_out_max
        self.RZ_out_ideal = RZ_out_ideal

        self.ES_ramp_steps = ES_ramp_steps
        self.ES_ramp_counter = 0
        self.ES_ramp_active = False
        self.ES_ramp_start_value = 0.0
        self.ES_ramp_end_value = 0.0

        self.ES_rain_conditions = False
        self.RZ_rain_conditions = False

        self.ES_setting = []
        self.RZ_setting = []
        self.setting_time = []

    def simulation_steps(self):
        for step in self.sim:
            self.handle_virtual_storage()
            self.handle_c_119_flows()
            self.real_time_control()
            self.track_control_settings()

        self.save_settings()

    def track_control_settings(self):
        self.ES_setting.append(
            self.links["P_eindhoven_out"].target_setting
            # * RzPumpCurve.interpolated_curve(self.nodes["pipe_ES"].depth)
        )
        self.RZ_setting.append(
            self.links["P_riool_zuid_out"].target_setting
            # * RzPumpCurve.interpolated_curve(self.nodes["pre_ontvangstkelder"].depth)
        )
        self.setting_time.append(self.sim.current_time)

    def real_time_control(self):
        self.get_forecast()
        self.set_storage()
        self.get_and_set_state()
        self.dwf_logic()
        self.wwf_logic()

    def get_and_set_state(self):
        ES_rain_forecast, RZ_rain_forecast = self.rain_predicted(4, 0.5, 0.5)

        if ES_rain_forecast or self.ES_rain_conditions:
            self.ES_rain_conditions = ES_rain_forecast or (
                self.ES_storage.stored_volume > 10_000  # Daily diff max + 3k margin
            )
        else:
            self.ES_rain_conditions = False

        if RZ_rain_forecast or self.RZ_rain_conditions:
            self.RZ_rain_conditions = ES_rain_forecast or (
                self.RZ_storage.stored_volume > 8_000  # Daily diff max + 3k margin
            )
        else:
            self.RZ_rain_conditions = False

    def dwf_logic(self):
        if not self.ES_rain_conditions:
            self.ES_ramp_active = False
            self.ES_ramp_counter = 0
            self.links["P_eindhoven_out"].target_setting = (
                self.ES_out_ideal / self.ES_out_max
            )
        if not self.RZ_rain_conditions:
            self.links["P_riool_zuid_out"].target_setting = (
                self.RZ_out_ideal / self.RZ_out_max
            )

    def wwf_logic(self):
        if self.ES_rain_conditions:
            if not self.ES_ramp_active:
                self.ES_ramp_active = True
                self.ES_ramp_counter = 0
                self.ES_ramp_start_value = self.links["P_eindhoven_out"].target_setting
                self.ES_ramp_end_value = (
                    EsPumpCurve.interpolated_curve(self.nodes["pipe_ES"].depth)
                    / self.ES_out_max
                )

            if self.ES_ramp_counter < self.ES_ramp_steps:
                increment = (
                    self.ES_ramp_end_value - self.ES_ramp_start_value
                ) / self.ES_ramp_steps
                self.links["P_eindhoven_out"].target_setting = (
                    self.ES_ramp_start_value + increment * self.ES_ramp_counter
                )
                self.ES_ramp_counter += 1
            else:
                self.links["P_eindhoven_out"].target_setting = self.ES_ramp_end_value
        if self.RZ_rain_conditions:
            self.links["P_riool_zuid_out"].target_setting = (
                EsPumpCurve.interpolated_curve(self.nodes["pre_ontvangstkelder"].depth)
                / self.RZ_out_max
            )

    def set_storage(self):
        self.ES_storage.update_stored_volume(self.nodes["pipe_ES"].volume)
        self.RZ_storage.update_stored_volume(self.RZ_storage.get_volume(self.links))

    def get_forecast(self):
        time = self.sim.current_time.replace(minute=0, second=0, microsecond=0)

        if time.hour % 6 == 0 and self.last_forecast_time != time:
            end_time = time + dt.timedelta(hours=48)
            self.current_forecast = (
                self.precipitation_forecast.loc[time:end_time].resample("h").sum()
            )
            self.last_forecast_time = time

    def rain_predicted(self, horizon, es_threshold, rz_threshold):
        time = self.sim.current_time.replace(minute=0, second=0, microsecond=0)
        end_time = time + dt.timedelta(hours=horizon)

        ES_forecast = self.current_forecast.loc[time:end_time, "ES"].sum()
        RZ_forecast = (
            self.current_forecast.loc[time:end_time, ["GE", "RZ1", "RZ2"]].sum().sum()
        )
        return ES_forecast > es_threshold, RZ_forecast > rz_threshold

    def save_settings(self):
        df = pd.DataFrame(
            {
                "ES_setting CMS": np.array(self.ES_setting),
                "RZ_setting CMS": np.array(self.RZ_setting),
            },
            index=self.setting_time,
        )

        df.to_csv("swmm_output/target_setting.csv")


if __name__ == "__main__":
    MODEL_NAME = "model_jip"
    simulation = RealTimeControl(
        model_path=rf"data\SWMM\{MODEL_NAME}.inp",
        step_size=300,
        report_start=dt.datetime(year=2024, month=7, day=1),
        start_time=dt.datetime(year=2024, month=7, day=1),
        end_time=dt.datetime(year=2024, month=7, day=10),
        virtual_pump_max=10,
    )
    simulation.start_simulation()

    from postprocess import PostProcess

    suffix = ""
    postprocess = PostProcess(model_name=MODEL_NAME)
    postprocess.plot_pumps(
        save=False, plot_rain=True, target_setting=True, suffix=suffix, storage=True
    )
