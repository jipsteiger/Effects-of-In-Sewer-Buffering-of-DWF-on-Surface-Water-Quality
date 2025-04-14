import pyswmm as ps
import datetime as dt
import pandas as pd
import numpy as np
import logging
from pump_curves import EsPumpCurve, RzPumpCurve
from simulation import Simulation
from storage import Storage, RZ_storage

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename="rtc_log.txt",
    filemode="w",
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

        self.ES_storage = Storage(11000)
        self.RZ_storage = RZ_storage(7500)

        self.ES_out_max = ES_out_max
        self.ES_out_ideal = ES_out_ideal
        self.RZ_out_max = RZ_out_max
        self.RZ_out_ideal = RZ_out_ideal

        self.ES_dwf = False
        self.RZ_dwf = False

        self.ES_setting = []
        self.RZ_setting = []
        self.setting_time = []

        self.ES_ramp_counter = 0
        self.ES_ramp_active = False
        self.ES_ramp_start_value = 0.0
        self.ES_ramp_end_value = 0.0
        self.ES_ramp_steps = 108  # or make this configurable
        self.ES_wwf_linger = False
        self.ES_ramp_active_back = False
        self.ES_ramp_counter_back = 0
        self.ES_transition_finished_back = False
        self.ES_ramp_start_value_back = 0
        self.ES_ramp_end_value_back = 0

        self.RZ_ramp_counter = 0
        self.RZ_ramp_active = False
        self.RZ_ramp_start_value = 0.0
        self.RZ_ramp_end_value = 0.0
        self.RZ_ramp_active_back = False
        self.RZ_ramp_counter_back = 0
        self.RZ_transition_finished_back = False
        self.RZ_ramp_start_value_back = 0.0
        self.RZ_ramp_end_value_back = 0.0
        self.RZ_ramp_steps = 108  # or make this configurable
        self.RZ_wwf_linger = False

        self.ES_predicted = False
        self.RZ_predicted = False
        self.ES_transition_finished = False
        self.RZ_transition_finished = False

    def simulation_steps(self):
        for step in self.sim:
            self.handle_virtual_storage()
            self.handle_c_119_flows()
            self.handle_geldrop_out_flows()

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
        self.orchestrate_rtc()

    def orchestrate_rtc(self):
        st_ES_predicted, st_RZ_predicted = self.rain_predicted(0, 6, 1, 3 * 1)

        if 10 <= self.sim.current_time.hour <= 23:
            lt_start = 6

            future_hour = (self.sim.current_time + dt.timedelta(hours=12)).hour

            if future_hour < 6 or future_hour > 20:
                lt_end = 12
            else:
                lt_end = 12 - (future_hour - 6)

            lt_ES_predicted, lt_RZ_predicted = self.rain_predicted(
                lt_start, lt_end, 1, 3 * 1
            )

            self.ES_predicted = st_ES_predicted or lt_ES_predicted or self.ES_predicted
            self.RZ_predicted = st_RZ_predicted or lt_RZ_predicted or self.RZ_predicted
        else:
            self.ES_predicted = st_ES_predicted
            self.RZ_predicted = st_RZ_predicted

        ES_raining, RZ_raining = self.is_raining(1, 3 * 1)

        ES_dwf = not self.ES_predicted and not ES_raining
        RZ_dwf = not self.RZ_predicted and not RZ_raining

        ES_transition_to_wwf = self.ES_predicted
        RZ_transition_to_wwf = self.RZ_predicted

        ES_wwf = ES_raining or self.ES_predicted
        RZ_wwf = RZ_raining or self.RZ_predicted
        if ES_dwf and not self.ES_wwf_linger:
            self.ES_dwf_logic()
            self.ES_transition_finished = False
            self.ES_transition_finished_back = False
        elif ES_transition_to_wwf and not self.ES_transition_finished:
            self.ES_transition_to_wwf_logic()
        elif ES_wwf or self.ES_transition_finished or self.ES_wwf_linger:
            self.ES_wwf_linger = True
            self.ES_wwf_logic()

        if ((self.ES_storage.stored_volume / self.ES_storage.V_max) < 1.25) and not (
            ES_wwf
        ):
            self.ES_wwf_linger = False

        if RZ_dwf and not self.RZ_wwf_linger:
            self.RZ_dwf_logic()
            self.RZ_transition_finished = False
            self.RZ_transition_finished_back = False
        elif RZ_transition_to_wwf and not self.RZ_transition_finished:
            self.RZ_transition_to_wwf_logic()
        elif RZ_wwf or self.RZ_transition_finished or self.RZ_wwf_linger:
            self.RZ_wwf_linger = True
            self.RZ_wwf_logic()

        if (
            (self.RZ_storage.stored_volume / self.RZ_storage.V_max) < 1.25
        ) and not RZ_wwf:
            self.RZ_wwf_linger = False

    def ES_dwf_logic(self):

        self.ES_ramp_active_back = False
        self.ES_ramp_counter_back = 0
        self.ES_ramp_active = False
        self.ES_ramp_counter = 0
        multiplier = max(self.ES_storage.stored_volume / self.ES_storage.V_max, 1)
        self.links["P_eindhoven_out"].target_setting = (
            self.ES_out_ideal / self.ES_out_max * multiplier
        )

    def RZ_dwf_logic(self):
        self.RZ_ramp_active_back = False
        self.RZ_ramp_counter_back = 0
        self.RZ_ramp_active = False
        self.RZ_ramp_counter = 0

        multiplier = max(self.RZ_storage.stored_volume / self.RZ_storage.V_max, 1)
        self.links["P_riool_zuid_out"].target_setting = (
            self.RZ_out_ideal / self.RZ_out_max * multiplier**2
        )

    def ES_transition_to_wwf_logic(self):
        if not self.ES_ramp_active:
            self.ES_ramp_active = True
            self.ES_ramp_counter = 0
            self.ES_ramp_start_value = self.ES_out_ideal / self.ES_out_max
            self.ES_ramp_end_value = (self.ES_out_ideal / self.ES_out_max) * 2
        if self.ES_ramp_counter < self.ES_ramp_steps:
            increment = (
                self.ES_ramp_end_value - self.ES_ramp_start_value
            ) / self.ES_ramp_steps
            self.links["P_eindhoven_out"].target_setting = (
                self.ES_ramp_start_value + increment * self.ES_ramp_counter
            )
            self.ES_ramp_counter += 1
            self.ES_ramp_start_value_back = (
                self.ES_ramp_start_value + increment * self.ES_ramp_counter
            )

            if self.ES_storage.FD() < 0.1:
                self.ES_transition_finished = True
            else:
                self.ES_transition_finished = False
        else:
            self.ES_transition_finished = True
            self.ES_wwf_logic()

    def RZ_transition_to_wwf_logic(self):
        if not self.RZ_ramp_active:
            self.RZ_ramp_active = True
            self.RZ_ramp_counter = 0
            self.RZ_ramp_start_value = self.RZ_out_ideal / self.RZ_out_max
            self.RZ_ramp_end_value = (
                RzPumpCurve.interpolated_curve(self.nodes["pre_ontvangstkelder"].depth)
                / self.RZ_out_max
            )

        if self.RZ_ramp_counter < self.RZ_ramp_steps:
            increment = (
                self.RZ_ramp_end_value - self.RZ_ramp_start_value
            ) / self.RZ_ramp_steps
            ramp_value = self.RZ_ramp_start_value + increment * self.RZ_ramp_counter
            self.links["P_riool_zuid_out"].target_setting = ramp_value

            self.RZ_ramp_counter += 1
            self.RZ_ramp_start_value_back = ramp_value  # Save for backward ramping
            self.RZ_transition_finished = False
        else:
            self.links["P_riool_zuid_out"].target_setting = self.RZ_ramp_end_value
            self.RZ_transition_finished = True
            self.RZ_wwf_logic()

    def ES_wwf_logic(self):
        self.ES_ramp_active = False
        self.ES_ramp_counter = 0

        current_ratio = (
            EsPumpCurve.interpolated_curve(self.nodes["pipe_ES"].depth)
            / self.ES_out_max
        )

        if (
            current_ratio < self.ES_ramp_end_value
        ) and not self.ES_transition_finished_back:
            if not self.ES_ramp_active_back:
                self.ES_ramp_active_back = True
                self.ES_ramp_counter_back = 0

            self.ES_ramp_end_value_back = current_ratio

            if self.ES_ramp_counter_back < self.ES_ramp_steps:
                increment = (
                    self.ES_ramp_end_value_back - self.ES_ramp_start_value_back
                ) / self.ES_ramp_steps
                ramp_value = (
                    self.ES_ramp_start_value_back
                    + increment * self.ES_ramp_counter_back
                )

                if current_ratio < ramp_value:
                    self.links["P_eindhoven_out"].target_setting = ramp_value
                else:
                    self.links["P_eindhoven_out"].target_setting = current_ratio

                self.ES_ramp_counter_back += 1
                if self.ES_storage.FD() < 0.1:
                    self.ES_transition_finished_back = True
                    self.ES_wwf_logic()
                else:
                    self.ES_transition_finished_back = False
            else:
                # If ramp is done, just use the current ratio
                self.links["P_eindhoven_out"].target_setting = current_ratio
                self.ES_transition_finished_back = True
        else:
            self.ES_ramp_active_back = False
            self.ES_ramp_counter_back = 0
            self.links["P_eindhoven_out"].target_setting = current_ratio

    def RZ_wwf_logic(self):
        self.RZ_ramp_active = False
        self.RZ_ramp_counter = 0

        current_ratio = (
            RzPumpCurve.interpolated_curve(self.nodes["pre_ontvangstkelder"].depth)
            / self.RZ_out_max
        )

        if (
            current_ratio < self.RZ_ramp_end_value
        ) and not self.RZ_transition_finished_back:
            if not self.RZ_ramp_active_back:
                self.RZ_ramp_active_back = True
                self.RZ_ramp_counter_back = 0

            self.RZ_ramp_end_value_back = current_ratio

            if self.RZ_ramp_counter_back < self.RZ_ramp_steps:
                increment = (
                    self.RZ_ramp_end_value_back - self.RZ_ramp_start_value_back
                ) / self.RZ_ramp_steps
                ramp_value = (
                    self.RZ_ramp_start_value_back
                    + increment * self.RZ_ramp_counter_back
                )

                if current_ratio < ramp_value:
                    self.links["P_riool_zuid_out"].target_setting = ramp_value
                else:
                    self.links["P_riool_zuid_out"].target_setting = current_ratio

                self.RZ_ramp_counter_back += 1
                if self.RZ_storage.FD() < 0.1:
                    self.RZ_transition_finished_back = True
                    self.RZ_wwf_logic()
                else:
                    self.RZ_transition_finished_back = False
            else:
                self.links["P_riool_zuid_out"].target_setting = current_ratio
                self.RZ_transition_finished_back = True
        else:
            self.RZ_transition_finished_back = True
            self.RZ_ramp_active_back = False
            self.RZ_ramp_counter_back = 0
            self.links["P_riool_zuid_out"].target_setting = current_ratio

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

    def rain_predicted(self, start_horizon, end_horizon, es_threshold, rz_threshold):
        time = self.sim.current_time.replace(minute=0, second=0, microsecond=0)
        start_time = time + dt.timedelta(hours=start_horizon)
        end_time = time + dt.timedelta(hours=end_horizon)

        ES_forecast = self.current_forecast.loc[start_time:end_time, "ES"].mean()
        RZ_forecast = (
            self.current_forecast.loc[start_time:end_time, ["GE", "RZ1", "RZ2"]]
            .sum()
            .mean()
        )
        return ES_forecast > es_threshold, RZ_forecast > rz_threshold

    def is_raining(self, es_threshold, rz_threshold):
        time = self.sim.current_time.replace(minute=0, second=0, microsecond=0)
        end_time = time + dt.timedelta(minutes=10)
        ES_raining = self.current_forecast.loc[time:end_time, "ES"].mean()
        RZ_raining = (
            self.current_forecast.loc[time:end_time, ["GE", "RZ1", "RZ2"]].sum().mean()
        )
        return ES_raining > es_threshold, RZ_raining > rz_threshold

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
        report_start=dt.datetime(year=2024, month=1, day=1),
        start_time=dt.datetime(year=2024, month=1, day=1),
        end_time=dt.datetime(year=2024, month=12, day=31),
        virtual_pump_max=10,
    )
    simulation.start_simulation()

    from postprocess import PostProcess

    suffix = ""
    postprocess = PostProcess(model_name=MODEL_NAME)
    postprocess.plot_pumps(
        save=False, plot_rain=True, target_setting=True, suffix=suffix, storage=True
    )
