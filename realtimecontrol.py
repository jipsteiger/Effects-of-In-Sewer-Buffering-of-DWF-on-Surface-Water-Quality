import datetime as dt
import pandas as pd
import numpy as np
import logging
from data.pump_curves import EsPumpCurve, RzPumpCurve
from simulation import Simulation
from storage import Storage, RZ_storage
from scipy.interpolate import interp1d
from typing import Callable, Tuple

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
        virtual_pump_max=10,
        constant_outflow=False,
        use_ensemble_forecast=False,
        ES_threshold=1,
        RZ_threshold=3,
        ES_certainty_threshold=0.9,
        RZ_certainty_threshold=0.9,
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
            constant_outflow,
        )
        self.current_forecast = None
        self.last_forecast_time = None
        self.use_ensemble_forecast = use_ensemble_forecast
        self.ES_threshold = ES_threshold
        self.RZ_threshold = RZ_threshold
        self.ES_certainty_threshold = ES_certainty_threshold
        self.RZ_certainty_threshold = RZ_certainty_threshold

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

        self.ES_last_setting = ES_out_ideal
        self.ES_wwf_linger = False

        self.RZ_last_setting = RZ_out_ideal
        self.RZ_wwf_linger = False

        self.ES_predicted = False
        self.RZ_predicted = False
        self.ES_transition_finished = False
        self.RZ_transition_finished = False

        self.ES_flow_OF = []
        self.ES_FD_OF = []
        self.RZ_flow_OF = []
        self.RZ_FD_OF = []
        self.ES_flow_OF_always = []
        self.RZ_flow_OF_always = []

        self.ES_state = None
        self.RZ_state = None
        self.ES_states = []
        self.RZ_states = []

        self.forecasts = read_forecasts()

    def simulation_steps(self):
        for _ in self.sim:
            self.handle_virtual_storage()
            self.handle_c_119_flows()
            self.handle_geldrop_out_flows()
            self.update_WQ()

            self.concentrations()

            self.set_storage_for_FD()
            self.real_time_control()
            self.track_control_settings()
            self.track_state()

        self.save_concentrations(name="RTC")
        self.save_settings()

    def track_control_settings(self):
        self.ES_setting.append(self.links["P_eindhoven_out"].target_setting)
        self.RZ_setting.append(self.links["P_riool_zuid_out"].target_setting)
        self.setting_time.append(self.sim.current_time)

    def track_state(self):
        self.ES_states.append(self.ES_state)
        self.RZ_states.append(self.RZ_state)

    def real_time_control(self):
        self.set_storage()
        if self.use_ensemble_forecast:
            self.get_current_forecasts()
            time = self.sim.current_time
            current_hour = time.replace(minute=0, second=0, microsecond=0)
            # As ensemble predictions are only in hour intervals, only need to check once per hour
            # Current forecast can be empty, therefore we immideatly return if this is the case, mainting previous forecast
            if (
                getattr(self, "last_rain_check_hour", None) != current_hour
            ) and not len(self.current_forecast) == 0:
                self.last_rain_check_hour = current_hour
                self.is_rain_predicted(
                    rain_predicted_func=self.rain_predicted_by_ensembles,
                    st_lb=0,
                    st_ub=6,
                    lt_lb=6,
                    lt_up=12,
                    ES_threshold=self.ES_threshold,
                    RZ_threshold=self.RZ_threshold,
                )
        else:
            self.get_current_ideal_forecast()
            self.is_rain_predicted(
                rain_predicted_func=self.rain_predicted,
                st_lb=0,
                st_ub=6,
                lt_lb=6,
                lt_up=12,
                ES_threshold=1,
                RZ_threshold=3,
            )

        self.orchestrate_rtc()
        self.ES_last_setting = self.links["P_eindhoven_out"].target_setting
        self.RZ_last_setting = self.links["P_riool_zuid_out"].target_setting

    def is_rain_predicted(
        self,
        rain_predicted_func: Callable[..., Tuple[bool, bool]],
        st_lb,
        st_ub,
        lt_lb,
        lt_up,
        ES_threshold,
        RZ_threshold,
    ):
        st_ES_predicted, st_RZ_predicted = rain_predicted_func(
            st_lb, st_ub, ES_threshold, RZ_threshold
        )

        if 11 <= self.sim.current_time.hour <= 23:
            lt_start = lt_lb

            future_hour = (self.sim.current_time + dt.timedelta(hours=12)).hour

            if future_hour < 6 or future_hour > 20:
                lt_end = lt_up
            else:
                lt_end = lt_up - (future_hour - lt_lb)

            lt_ES_predicted, lt_RZ_predicted = rain_predicted_func(
                lt_start, lt_end, ES_threshold, RZ_threshold
            )

            self.ES_predicted = st_ES_predicted or lt_ES_predicted or self.ES_predicted
            self.RZ_predicted = st_RZ_predicted or lt_RZ_predicted or self.RZ_predicted
        else:
            self.ES_predicted = st_ES_predicted
            self.RZ_predicted = st_RZ_predicted

    def orchestrate_rtc(self):
        ES_raining, RZ_raining = self.is_raining(2, 3 * 1)

        ES_dwf = not self.ES_predicted and not ES_raining
        RZ_dwf = not self.RZ_predicted and not RZ_raining

        ES_transition_to_wwf = self.ES_predicted
        RZ_transition_to_wwf = self.RZ_predicted

        ES_wwf = ES_raining or self.ES_predicted
        RZ_wwf = RZ_raining or self.RZ_predicted

        if ES_dwf and not self.ES_wwf_linger:
            self.ES_dwf_logic()
            self.ES_state = "dwf"
        elif ES_transition_to_wwf and not (
            self.nodes["pipe_ES"].total_inflow > self.ES_out_ideal * 2
        ):
            self.ES_transition_to_wwf_logic()
            self.ES_state = "transition"
        elif ES_wwf or self.ES_wwf_linger:
            self.ES_wwf_linger = True
            self.ES_wwf_logic()
            self.ES_state = "wwf"
        if ((self.ES_storage.stored_volume / self.ES_storage.V_max) < 1.25) and not (
            ES_wwf
        ):
            self.ES_wwf_linger = False

        if RZ_dwf and not self.RZ_wwf_linger:
            self.RZ_dwf_logic()
            self.RZ_state = "dwf"
        elif (
            RZ_transition_to_wwf
            and not (
                self.nodes["Nod_112"].total_inflow + self.nodes["Nod_104"].total_inflow
            )
            > self.RZ_out_ideal * 2
        ):
            self.RZ_transition_to_wwf_logic()
            self.RZ_state = "transition"
        elif RZ_wwf or self.RZ_wwf_linger:
            self.RZ_wwf_linger = True
            self.RZ_wwf_logic()
            self.RZ_state = "wwf"

        if (
            (self.RZ_storage.stored_volume / self.RZ_storage.V_max) < 1.25
        ) and not RZ_wwf:
            self.RZ_wwf_linger = False

        self.create_OF_values(
            ES_dwf, ES_transition_to_wwf, ES_wwf, RZ_dwf, RZ_transition_to_wwf, RZ_wwf
        )

    def ES_dwf_logic(self):
        multiplier = max(self.ES_storage.stored_volume / self.ES_storage.V_max, 1)
        self.links["P_eindhoven_out"].target_setting = (
            self.ES_out_ideal / self.ES_out_max * multiplier
        )

    def RZ_dwf_logic(self):
        multiplier = max(self.RZ_storage.stored_volume / self.RZ_storage.V_max, 1)
        self.links["P_riool_zuid_out"].target_setting = (
            self.RZ_out_ideal / self.RZ_out_max * multiplier
        )

    def ES_transition_to_wwf_logic(self):
        inflow = self.nodes["pipe_ES"].total_inflow
        setting = max(self.ES_out_ideal, inflow)
        if (
            setting / self.ES_out_max
        ) / self.ES_last_setting > 1.025:  # Check if volume is not to low? When stops?
            self.links["P_eindhoven_out"].target_setting = self.ES_last_setting * 1.0125
        else:
            self.links["P_eindhoven_out"].target_setting = setting / self.ES_out_max

    def RZ_transition_to_wwf_logic(self):
        inflow = self.nodes["Nod_112"].total_inflow + self.nodes["Nod_104"].total_inflow
        setting = max(self.RZ_out_ideal, inflow)
        if (setting / self.RZ_out_max) / self.RZ_last_setting > 1.025:
            self.links["P_riool_zuid_out"].target_setting = (
                self.RZ_last_setting * 1.0125
            )
        else:
            self.links["P_riool_zuid_out"].target_setting = setting / self.RZ_out_max

    def ES_wwf_logic(self):
        current_ratio = (
            EsPumpCurve.interpolated_curve(self.nodes["pipe_ES"].depth)
            / self.ES_out_max
        )

        if current_ratio < self.ES_out_ideal / self.ES_out_max:
            self.links["P_eindhoven_out"].target_setting = (
                self.ES_out_ideal / self.ES_out_max
            )
        else:
            self.links["P_eindhoven_out"].target_setting = current_ratio

    def RZ_wwf_logic(self):
        current_ratio = (
            RzPumpCurve.interpolated_curve(self.nodes["pre_ontvangstkelder"].depth)
            / self.RZ_out_max
        )

        if current_ratio < self.RZ_out_ideal / self.RZ_out_max:
            self.links["P_riool_zuid_out"].target_setting = (
                self.RZ_out_ideal / self.RZ_out_max
            )
        else:
            self.links["P_riool_zuid_out"].target_setting = current_ratio

    def create_OF_values(
        self, ES_dwf, ES_transition_to_wwf, ES_wwf, RZ_dwf, RZ_transition_to_wwf, RZ_wwf
    ):
        self.ES_flow_OF_always.append(
            (self.links["P_eindhoven_out"].flow - self.ES_out_ideal) ** 2
            / self.ES_out_ideal
        )
        if (ES_dwf and not self.ES_wwf_linger) or (
            ES_transition_to_wwf
            and not (self.nodes["pipe_ES"].total_inflow > self.ES_out_ideal * 2)
        ):
            self.ES_flow_OF.append(
                (self.links["P_eindhoven_out"].flow - self.ES_out_ideal) ** 2
                / self.ES_out_ideal
            )
            self.ES_first_WWF_noted = False
        elif (ES_wwf or self.ES_wwf_linger) and not self.ES_first_WWF_noted:
            self.ES_FD_OF.append(self.ES_storage.FD())
            self.ES_first_WWF_noted = True

        self.RZ_flow_OF_always.append(
            (self.links["P_riool_zuid_out"].flow - self.RZ_out_ideal) ** 2
            / self.RZ_out_ideal
        )
        if (RZ_dwf and not self.RZ_wwf_linger) or (
            RZ_transition_to_wwf
            and not (
                (
                    self.nodes["Nod_112"].total_inflow
                    + self.nodes["Nod_104"].total_inflow
                )
                > self.RZ_out_ideal * 2
            )
        ):
            self.RZ_flow_OF.append(
                (self.links["P_riool_zuid_out"].flow - self.RZ_out_ideal) ** 2
                / self.RZ_out_ideal
            )
            self.RZ_first_WWF_noted = False
        elif (RZ_wwf or self.RZ_wwf_linger) and not self.RZ_first_WWF_noted:
            self.RZ_FD_OF.append(self.RZ_storage.FD())
            self.RZ_first_WWF_noted = True

    def get_OF(self):
        return (
            self.ES_flow_OF,
            self.ES_FD_OF,
            self.RZ_flow_OF,
            self.RZ_FD_OF,
            self.ES_flow_OF_always,
            self.RZ_flow_OF_always,
        )

    def get_state(self):
        return self.setting_time, self.ES_states, self.RZ_states

    def set_storage(self):
        self.ES_storage.update_stored_volume(self.nodes["pipe_ES"].volume)
        self.RZ_storage.update_stored_volume(self.RZ_storage.get_volume(self.links))

    def get_current_ideal_forecast(self):
        time = self.sim.current_time.replace(minute=0, second=0, microsecond=0)

        if time.hour % 6 == 0 and self.last_forecast_time != time:
            end_time = time + dt.timedelta(hours=48)
            self.current_forecast = self.precipitation_forecast.loc[time:end_time]
            self.last_forecast_time = time

    def get_current_forecasts(self, upperbound=24):
        time = self.sim.current_time.replace(minute=0, second=0, microsecond=0)
        if time.hour % 6 == 0 and self.last_forecast_time != time:
            upperbound_time = time + dt.timedelta(hours=upperbound)
            current_forecast = self.forecasts[
                (self.forecasts.date == time)
                & (self.forecasts.date_of_forecast <= upperbound_time)
            ]
            self.current_forecast = current_forecast.groupby(
                ["region", "date_of_forecast"]
            )["ensembles"].apply(lambda x: [item for sublist in x for item in sublist])
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

    def rain_predicted_by_ensembles(self, time_lb, time_ub, ES_threshold, RZ_threshold):
        time = self.sim.current_time.replace(minute=0, second=0, microsecond=0)
        start_time = time + dt.timedelta(
            hours=time_lb + 2
        )  # Window is delayed because no now-casts are made in the prediction (ie at 18:00 no prediction for 18:00 or 19:00)
        end_time = time + dt.timedelta(hours=time_ub + 2)
        current_forecast_window = self.current_forecast.loc[
            (
                self.current_forecast.index.get_level_values("date_of_forecast")
                >= start_time
            )
            & (
                self.current_forecast.index.get_level_values("date_of_forecast")
                <= end_time
            )
        ]

        if len(current_forecast_window) == 0:
            return self.ES_predicted, self.RZ_predicted

        quantile_values_by_region = {}
        for region_name in current_forecast_window.index.get_level_values(
            "region"
        ).unique():
            region_forecast_series = current_forecast_window.loc[region_name]
            quantile_values_by_region[region_name] = []
            for _, ensemble_values in region_forecast_series.items():
                ensemble_array = np.array(ensemble_values)
                if len(ensemble_array) == 0 or np.all(ensemble_array == 0):
                    quantile_values_by_region[region_name].append(0.0)
                    continue

                sorted_values = np.sort(ensemble_array)
                empirical_probabilities = np.linspace(0, 1, len(sorted_values))

                # Interpolate inverse CDF (quantile function)
                quantile_function = interp1d(
                    empirical_probabilities,
                    sorted_values,
                    bounds_error=False,
                    fill_value=(sorted_values[0], sorted_values[-1]),
                )
                if region_name == "ES":
                    certainty_threshold = self.ES_certainty_threshold
                else:
                    certainty_threshold = self.RZ_certainty_threshold

                # Append the rainfall value at the given confidence level
                # Here you calculate that with a certain confidence level, there will less rain that.
                # Thus: With 90% confidence, the rainfall will be less than X mm.
                quantile_values_by_region[region_name].append(
                    float(quantile_function(certainty_threshold))
                )

        ES_predicted = np.mean(quantile_values_by_region["ES"]) > ES_threshold

        RZ_totals = [
            np.mean(quantile_values_by_region[region])
            for region in ["RZ1", "RZ2", "GE"]
        ]
        RZ_predicted = any(total > RZ_threshold for total in RZ_totals) or np.mean(
            RZ_totals
        ) > (RZ_threshold / 3)

        return ES_predicted, RZ_predicted

    def is_raining(self, es_threshold, rz_threshold):
        time = self.sim.current_time.replace(minute=0, second=0, microsecond=0)
        end_time = time + dt.timedelta(minutes=10)
        ES_raining = self.precipitation_forecast.loc[time:end_time, "ES"].mean()
        RZ_raining = (
            self.precipitation_forecast.loc[time:end_time, ["GE", "RZ1", "RZ2"]]
            .sum()
            .mean()
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

        df.to_csv("output_swmm/target_setting.csv")


def read_forecasts():
    forecasts = pd.read_csv(
        rf"data\precipitation\csv_forecasts\forecast_data.csv", index_col=0
    )
    forecasts["date"] = pd.to_datetime(forecasts["date"])
    forecasts["date_of_forecast"] = pd.to_datetime(forecasts["date_of_forecast"])
    forecasts["ensembles"] = forecasts["ensembles"].apply(
        lambda s: [float(x) for x in s.strip("[]").split()]
    )
    return forecasts


if __name__ == "__main__":
    MODEL_NAME = "model_jip"
    simulation = RealTimeControl(
        model_path=rf"data\SWMM\{MODEL_NAME}.inp",
        step_size=300,
        report_start=dt.datetime(year=2024, month=7, day=1),
        start_time=dt.datetime(year=2024, month=7, day=1),
        end_time=dt.datetime(year=2024, month=7, day=31),
        virtual_pump_max=10,
        use_ensemble_forecast=True,
        ES_threshold=2.25,
        RZ_threshold=4,
        certainty_threshold=0.75,
    )
    simulation.start_simulation()
