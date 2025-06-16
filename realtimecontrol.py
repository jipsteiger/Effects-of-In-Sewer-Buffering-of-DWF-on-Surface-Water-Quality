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
        do_load_averaging=False,
        ES_rain_threshold=1,
        RZ_rain_threshold=1,
        ES_threshold=1,
        RZ_threshold=1,
        ES_certainty_threshold=0.9,
        RZ_certainty_threshold=0.9,
        ES_out_max: float = 3.888,
        ES_out_ideal: float = 0.663,
        RZ_out_max: float = 4.7222,
        RZ_out_ideal: float = 0.5218,
        ES_V_avg_target=8000,
        RZ_V_avg_target=5000,
        ES_target_avg_load=33.80,
        RZ_target_avg_load=24.2,
        Kp=0.0001,
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
        self.do_load_averaging = do_load_averaging
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

        self.ES_V_avg_target = ES_V_avg_target
        self.RZ_V_avg_target = RZ_V_avg_target
        self.ES_target_avg_load = ES_target_avg_load
        self.RZ_target_avg_load = RZ_target_avg_load
        self.Kp = Kp

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
                ES_threshold=self.ES_threshold,
                RZ_threshold=self.RZ_threshold,
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
            if not self.do_load_averaging:
                self.ES_dwf_logic_flow()
            else:
                self.ES_dwf_logic_load()
            self.ES_state = "dwf"
        elif ES_transition_to_wwf and not (
            self.nodes["pipe_ES"].total_inflow > self.ES_out_ideal * 2
        ):
            self.ES_transition_to_wwf_logic_flow()
            self.ES_state = "transition"
        elif ES_wwf or self.ES_wwf_linger:
            self.ES_wwf_linger = True
            self.ES_wwf_logic_flow()
            self.ES_state = "wwf"
        if ((self.ES_storage.stored_volume / self.ES_storage.V_max) < 1.25) and not (
            ES_wwf
        ):
            self.ES_wwf_linger = False

        if RZ_dwf and not self.RZ_wwf_linger:
            if not self.do_load_averaging:
                self.RZ_dwf_logic_flow()
            else:
                self.RZ_dwf_logic_load()
            self.RZ_state = "dwf"
        elif (
            RZ_transition_to_wwf
            and not (
                self.nodes["Nod_112"].total_inflow + self.nodes["Nod_104"].total_inflow
            )
            > self.RZ_out_ideal * 2
        ):
            self.RZ_transition_to_wwf_logic_flow()
            self.RZ_state = "transition"
        elif RZ_wwf or self.RZ_wwf_linger:
            self.RZ_wwf_linger = True
            self.RZ_wwf_logic_flow()
            self.RZ_state = "wwf"

        if (
            (self.RZ_storage.stored_volume / self.RZ_storage.V_max) < 1.25
        ) and not RZ_wwf:
            self.RZ_wwf_linger = False

        self.create_OF_values(
            ES_dwf, ES_transition_to_wwf, ES_wwf, RZ_dwf, RZ_transition_to_wwf, RZ_wwf
        )

    def ES_dwf_logic_flow(self):
        multiplier = max(self.ES_storage.stored_volume / self.ES_storage.V_max, 1)
        self.links["P_eindhoven_out"].target_setting = (
            self.ES_out_ideal / self.ES_out_max * multiplier
        )

    def ES_dwf_logic_load(self):
        V, concentrations = self.ESConcentrationStorage.get_current_state()
        logging.debug("##############################")
        logging.debug(self.sim.current_time)

        NH4_conc = concentrations["NH4_sew"]
        if NH4_conc > 0.1:
            Q_target = self.ES_target_avg_load / NH4_conc  # g/s / g/m3 = m3/s
            logging.debug(f"{Q_target=}")
        else:
            Q_target = self.ES_out_ideal
        logging.debug(f"{NH4_conc=}")
        volume_error = V - self.ES_V_avg_target
        logging.debug(f"{volume_error=}")
        Q_target_correction = self.Kp * volume_error
        Q_target_corrected = Q_target + Q_target_correction
        logging.debug(f"{Q_target_corrected=}")
        self.links["P_eindhoven_out"].target_setting = (
            max(0, Q_target_corrected) / self.ES_out_max
        )

    def RZ_dwf_logic_flow(self):
        multiplier = max(self.RZ_storage.stored_volume / self.RZ_storage.V_max, 1)
        self.links["P_riool_zuid_out"].target_setting = (
            self.RZ_out_ideal / self.RZ_out_max * multiplier
        )

    def RZ_dwf_logic_load(self):
        V, concentrations = self.RZConcentrationStorage.get_current_state()
        NH4_conc = concentrations["NH4_sew"]
        if NH4_conc > 0.1:
            Q_target = self.RZ_target_avg_load / NH4_conc
        else:
            Q_target = self.RZ_out_ideal
        volume_error = V - self.RZ_V_avg_target
        Q_target_correction = self.Kp * volume_error
        Q_target_corrected = Q_target + Q_target_correction
        self.links["P_riool_zuid_out"].target_setting = (
            max(0, Q_target_corrected) / self.RZ_out_max
        )

    def ES_transition_to_wwf_logic_flow(self):
        inflow = self.nodes["pipe_ES"].total_inflow
        setting = max(self.ES_out_ideal, inflow)
        if (
            setting / self.ES_out_max
        ) / self.ES_last_setting > 1.025:  # Check if volume is not to low? When stops?
            self.links["P_eindhoven_out"].target_setting = self.ES_last_setting * 1.0125
        else:
            self.links["P_eindhoven_out"].target_setting = setting / self.ES_out_max

    def RZ_transition_to_wwf_logic_flow(self):
        inflow = self.nodes["Nod_112"].total_inflow + self.nodes["Nod_104"].total_inflow
        setting = max(self.RZ_out_ideal, inflow)
        if (setting / self.RZ_out_max) / self.RZ_last_setting > 1.025:
            self.links["P_riool_zuid_out"].target_setting = (
                self.RZ_last_setting * 1.0125
            )
        else:
            self.links["P_riool_zuid_out"].target_setting = setting / self.RZ_out_max

    def ES_wwf_logic_flow(self):
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

    def RZ_wwf_logic_flow(self):
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

        # Area weights (make sure these sum to 1.0)
        area_factors = {"GE": 0.25, "RZ1": 0.375, "RZ2": 0.375}

        # Get mean forecast for each catchment
        rz_forecasts = self.current_forecast.loc[
            start_time:end_time, ["GE", "RZ1", "RZ2"]
        ].mean()

        # Check 1: Any individual catchment has intense rain
        per_catchment_check = any(
            (forecast / area_factors[cat]) > rz_threshold
            for cat, forecast in rz_forecasts.items()
        )

        # Check 2: Weighted sum of forecasts exceeds threshold
        weighted_sum = sum(
            area_factors[cat] * forecast for cat, forecast in rz_forecasts.items()
        )
        combined_rz_check = weighted_sum > rz_threshold

        # ES forecast as before
        ES_forecast = self.current_forecast.loc[start_time:end_time, "ES"].mean()
        es_check = ES_forecast > es_threshold

        return es_check, (combined_rz_check or per_catchment_check)

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
        area_factors = {"GE": 0.25, "RZ1": 0.375, "RZ2": 0.375}
        # RZ: Compute area-weighted average
        weighted_sum = sum(
            area_factors[region] * np.mean(quantile_values_by_region[region])
            for region in ["GE", "RZ1", "RZ2"]
        )
        per_catchment_check = any(
            (np.mean(quantile_values_by_region[region]) / area_factors[region])
            > RZ_threshold
            for region in ["GE", "RZ1", "RZ2"]
        )
        RZ_predicted = (weighted_sum > RZ_threshold) or per_catchment_check

        return ES_predicted, RZ_predicted

    def is_raining(self, es_threshold, rz_threshold):
        time = self.sim.current_time.replace(minute=0, second=0, microsecond=0)
        end_time = time + dt.timedelta(minutes=10)

        # Area contributions
        area_factors = {"GE": 0.25, "RZ1": 0.375, "RZ2": 0.375}

        # ES check
        ES_raining = self.precipitation_forecast.loc[time:end_time, "ES"].mean()
        es_check = ES_raining > es_threshold

        # Mean forecast over time for each catchment
        rz_means = self.precipitation_forecast.loc[
            time:end_time, ["GE", "RZ1", "RZ2"]
        ].mean()

        # Check 1: Individual catchment exceeds normalized threshold
        per_catchment_check = any(
            (forecast / area_factors[cat]) > rz_threshold
            for cat, forecast in rz_means.items()
        )

        # Check 2: Weighted sum of all catchments
        weighted_sum = sum(
            area_factors[cat] * forecast for cat, forecast in rz_means.items()
        )
        combined_rz_check = (weighted_sum > rz_threshold) or per_catchment_check

        return es_check, combined_rz_check

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
        end_time=dt.datetime(year=2024, month=7, day=10),
        virtual_pump_max=10,
        use_ensemble_forecast=True,
        do_load_averaging=True,
        ES_threshold=0.75,
        RZ_threshold=2.5,
        ES_certainty_threshold=0.7,
        RZ_certainty_threshold=0.925,
    )
    simulation.start_simulation()
    # timesteps, ES_states, RZ_states = simulation.get_state()
    from postprocess import PostProcess

    SUFFIX = "RTC_load"
    postprocess = PostProcess(model_name=MODEL_NAME)
    postprocess.create_outfall_txt_concentrate(suffix=SUFFIX, specific_version="RTC")

    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio

    df_west = pd.read_csv(
        r"output_swmm\latest_out_ES_out.csv",
        # rf'data\WEST\WEST_modelRepository\Model_Dommel_Full\wwtp_control.out.txt',
        delimiter=";",
        decimal=",",
        index_col=0,
        parse_dates=True,
    )

    fig = go.Figure()

    for key in df_west.keys():
        fig.add_trace(
            go.Scatter(
                x=df_west.index,
                y=df_west[key].astype(float),
                mode="lines",
                name=f"WEST {key}",
            )
        )

    df_west2 = pd.read_csv(
        r"output_swmm\06-04_11-40_out_ES_RTC.txt",
        # rf'data\WEST\WEST_modelRepository\Model_Dommel_Full\wwtp_control.out.txt',
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    start_date = pd.Timestamp("2024-01-01")
    df_west2["timestamp"] = start_date + pd.to_timedelta(
        df_west2.index.astype(float), unit="D"
    )
    df_west2.set_index("timestamp", inplace=True)

    for key in df_west2.keys():
        fig.add_trace(
            go.Scatter(
                x=df_west2.index,
                y=df_west2[key].astype(float),
                mode="lines",
                name=f"RTC base {key}",
            )
        )

    df_west3 = pd.read_csv(
        r"output_swmm\06-01_16-25_out_ES_No_RTC.txt",
        # rf'data\WEST\WEST_modelRepository\Model_Dommel_Full\wwtp_control.out.txt',
        delimiter="\t",
        header=0,
        index_col=0,
        low_memory=False,
    ).iloc[1:, :]
    start_date = pd.Timestamp("2024-01-01")
    df_west3["timestamp"] = start_date + pd.to_timedelta(
        df_west3.index.astype(float), unit="D"
    )
    df_west3.set_index("timestamp", inplace=True)

    for key in df_west3.keys():
        fig.add_trace(
            go.Scatter(
                x=df_west3.index,
                y=df_west3[key].astype(float),
                mode="lines",
                name=f"NO RTC base {key}",
            )
        )

    pio.show(fig, renderer="browser")

    a = np.mean(
        abs(df_west.loc["2024-07-08":"2024-07-09", "NH4_sew"].values.astype(float))
    ) / (24 * 60 * 60)
    b = np.mean(
        abs(df_west2.loc["2024-07-08":"2024-07-09", "NH4_sew"].values.astype(float))
    ) / (24 * 60 * 60)
    c = np.mean(
        abs(df_west3.loc["2024-07-08":"2024-07-09", "NH4_sew"].values.astype(float))
    ) / (24 * 60 * 60)
    print(a, b, c)
