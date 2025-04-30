import pandas as pd


class EmpericalSewerWQ:
    def __init__(
        self,
        # Dictionary of concentrations
        concentration_dict,
        # Events
        Q_event=96000,  # [m3/d] Flow value start of event
        Qsw_event8=12000,  # [m3/d] Flow sw value for start event 8
        Qsw_event9=96000,  # [m3/d] Flow sw value for start event 9
        FD_event=0.21,  # [-] Filling degree value for start event 3
        T_Average=0.02083,  # [d] Averaging time for Q_mean
        # Averages
        Q_95_av=70000,  # [m3/d] Daily average Q, 95%ile
        COD_av=642,  # [g/m3] COD daily average concentration
        CODs_av=197.3,  # [g/m3] CODs daily average concentration
        TSS_av=289.6,  # [g/m3] TSS daily average concentration
        NH4_av=35,  # [g/m3] NH4 daily average concentration
        PO4_av=7,  # [g/m3] PO4 daily average concentration
        # proc2
        alpha_COD=0.8,  # [-] Alpha for COD
        alpha_CODs=1,  # [-] Alpha for CODs
        alpha_TSS=1,  # [-] Alpha for TSS
        alpha_NH4=1,  # [-] Alpha for NH4
        alpha_PO4=1,  # [-] Alpha for PO4
        # proc3
        beta_COD=0.8,  # [-] Beta for COD
        beta_CODs=1,  # [-] Beta for CODs
        beta_TSS=1,  # [-] Beta for TSS
        beta_NH4=1,  # [-] Beta for NH4
        beta_PO4=1,  # [-] Beta for PO4
        # proc4
        proc4_slope1_COD=0.576,  # [-] Slope1 for COD
        proc4_slope1_CODs=0.576,  # [-] Slope1 for CODs
        proc4_slope1_TSS=0.576,  # [-] Slope1 for TSS
        proc4_slope1_NH4=0.576,  # [-] Slope1 for NH4
        proc4_slope1_PO4=0.576,  # [-] Slope1 for PO4
        proc4_slope2_COD=0.288,  # [-] Slope2 for COD
        proc4_slope2_CODs=0.288,  # [-] Slope2 for CODs
        proc4_slope2_TSS=0.288,  # [-] Slope2 for TSS
        proc4_slope2_NH4=0.288,  # [-] Slope2 for NH4
        proc4_slope2_PO4=0.288,  # [-] Slope2 for PO4
        window_proc4_COD=0.8333,  # [d] Window length for COD
        window_proc4_CODs=0.8333,  # [d] Window length for CODs
        window_proc4_TSS=0.8333,  # [d] Window length for TSS
        window_proc4_NH4=0.8333,  # [d] Window length for NH4
        window_proc4_PO4=0.8333,  # [d] Window length for PO4
        # proc5
        proc5_slope_CODs=0.576,  # [-] Slope for CODs
        proc5_slope_NH4=0.576,  # [-] Slope for NH4
        proc5_slope_PO4=0.576,  # [-] Slope for PO4
        window_proc5_CODs=1.0833,  # [d] Window length for CODs
        window_proc5_NH4=1.0833,  # [d] Window length for NH4
        window_proc5_PO4=1.0833,  # [d] Window length for PO4
        # proc6
        Q_proc6=96000,  # [m3/d] Flow value for start of event
        peak_COD_high=600,  # [g/m3] High COD peak concentration
        peak_COD_low=300,  # [g/m3] Low COD peak concentration
        proc6_slope1_COD=1728,  # [-] Slope1 for COD
        proc6_slope2_COD=1296,  # [-] Slope2 for COD
        peak_TSS_high=200,  # [g/m3] High TSS peak concentration
        peak_TSS_low=100,  # [g/m3] Low TSS peak concentration
        proc6_slope1_TSS=1728,  # [-] Slope1 for TSS
        proc6_slope2_TSS=1296,  # [-] Slope2 for TSS
        proc6_t1_COD=0.5,  # [d] Time1 for COD high
        proc6_t2_COD=1.5,  # [d] Time2 for COD low
        proc6_t1_TSS=0.5,  # [d] Time1 for TSS high
        proc6_t2_TSS=1.5,  # [d] Time2 for TSS low
        # proc7
        proc7_slope1_COD=2880,  # [-] Slope1 for COD
        proc7_slope2_COD=720,  # [-] Slope2 for COD
        proc7_slope1_TSS=2880,  # [-] Slope1 for TSS
        proc7_slope2_TSS=720,  # [-] Slope2 for TSS
    ):
        # Assign all passed parameters to instance variables
        self.concentration_dict = concentration_dict

        self.Q_event = Q_event
        self.Qsw_event8 = Qsw_event8
        self.Qsw_event9 = Qsw_event9
        self.FD_event = FD_event
        self.T_Average = T_Average

        self.Q_95_av = Q_95_av
        self.COD_av = COD_av
        self.CODs_av = CODs_av
        self.TSS_av = TSS_av
        self.NH4_av = NH4_av
        self.PO4_av = PO4_av

        self.alpha_COD = alpha_COD
        self.alpha_CODs = alpha_CODs
        self.alpha_TSS = alpha_TSS
        self.alpha_NH4 = alpha_NH4
        self.alpha_PO4 = alpha_PO4

        self.beta_COD = beta_COD
        self.beta_CODs = beta_CODs
        self.beta_TSS = beta_TSS
        self.beta_NH4 = beta_NH4
        self.beta_PO4 = beta_PO4

        self.proc4_slope1_COD = proc4_slope1_COD
        self.proc4_slope1_CODs = proc4_slope1_CODs
        self.proc4_slope1_TSS = proc4_slope1_TSS
        self.proc4_slope1_NH4 = proc4_slope1_NH4
        self.proc4_slope1_PO4 = proc4_slope1_PO4

        self.proc4_slope2_COD = proc4_slope2_COD
        self.proc4_slope2_CODs = proc4_slope2_CODs
        self.proc4_slope2_TSS = proc4_slope2_TSS
        self.proc4_slope2_NH4 = proc4_slope2_NH4
        self.proc4_slope2_PO4 = proc4_slope2_PO4

        self.window_proc4_COD = window_proc4_COD
        self.window_proc4_CODs = window_proc4_CODs
        self.window_proc4_TSS = window_proc4_TSS
        self.window_proc4_NH4 = window_proc4_NH4
        self.window_proc4_PO4 = window_proc4_PO4

        self.proc5_slope_CODs = proc5_slope_CODs
        self.proc5_slope_NH4 = proc5_slope_NH4
        self.proc5_slope_PO4 = proc5_slope_PO4

        self.window_proc5_CODs = window_proc5_CODs
        self.window_proc5_NH4 = window_proc5_NH4
        self.window_proc5_PO4 = window_proc5_PO4

        self.Q_proc6 = Q_proc6
        self.peak_COD_high = peak_COD_high
        self.peak_COD_low = peak_COD_low
        self.proc6_slope1_COD = proc6_slope1_COD
        self.proc6_slope2_COD = proc6_slope2_COD
        self.peak_TSS_high = peak_TSS_high
        self.peak_TSS_low = peak_TSS_low
        self.proc6_slope1_TSS = proc6_slope1_TSS
        self.proc6_slope2_TSS = proc6_slope2_TSS
        self.proc6_t1_COD = proc6_t1_COD
        self.proc6_t2_COD = proc6_t2_COD
        self.proc6_t1_TSS = proc6_t1_TSS
        self.proc6_t2_TSS = proc6_t2_TSS

        self.proc7_slope1_COD = proc7_slope1_COD
        self.proc7_slope2_COD = proc7_slope2_COD
        self.proc7_slope1_TSS = proc7_slope1_TSS
        self.proc7_slope2_TSS = proc7_slope2_TSS

        self.state = self.set_initial_state()
        self.previous_state = self.state.copy()

        self.log = {
            "H2O_sew": [],
            "COD_part": [],
            "COD_sol": [],
            "X_TSS_sew": [],
            "NH4_sew": [],
            "PO4_sew": [],
        }
        self.index = []

    def set_initial_state(self):
        state = {}

        state["Q_integral"] = 0
        state["Q_mean"] = 0

        state["event"] = 0
        state["event8"] = 0
        state["event8_h"] = 0

        state["proc2_NH4"] = 1
        state["proc3_NH4"] = 1
        state["proc4_NH4"] = 1
        state["proc5_NH4"] = 1
        state["proc2_CODs"] = 1
        state["proc3_CODs"] = 1
        state["proc4_CODs"] = 1
        state["proc5_CODs"] = 1
        state["proc2_PO4"] = 1
        state["proc3_PO4"] = 1
        state["proc4_PO4"] = 1
        state["proc5_PO4"] = 1
        state["proc2_COD"] = 1
        state["proc3_COD"] = 1
        state["proc4_COD"] = 1
        state["proc2_TSS"] = 1
        state["proc3_TSS"] = 1
        state["proc4_TSS"] = 1

        return state

    def write_output_log(self, location=""):
        output = pd.DataFrame(self.log)
        output.to_csv(rf"effluent_concentration/{location}.Effluent.csv")

    def get_latest_log(self):
        return {
            key: (values[-1] if values else None) for key, values in self.log.items()
        }  # should be values in g/d

    def update(self, t, inflow_H2O_sew, FD):
        """_summary_

        Args:
            t (_type_): _description_
            inflow_H2O_sew (_type_): [m3/d]
            FD (_type_): Tank filling degree
        """
        dt = (
            t - self.previous_state.get("t", t)
        ).total_seconds() / 86400  # Should return decimal day
        if dt <= 0:
            dt = 1e-6  # prevent division by zero

        # inflow
        Q_in = inflow_H2O_sew
        self.state["Q_in"] = Q_in

        # Q_integral
        self.state["Q_integral"] += Q_in * dt

        hour_key = f"H_{int(t.hour)}"

        # DWF upper bound
        Q_DWF_UB = self.Q_95_av * getattr(
            self.concentration_dict["Q_95_norm"], hour_key
        )
        self.state["Q_DWF_UB"] = Q_DWF_UB

        # Storm water
        Qsw = max(0, Q_in - Q_DWF_UB)
        self.state["Qsw"] = Qsw

        # Q_mean
        self.state["Q_mean"] += (Q_in - self.state["Q_mean"]) / self.T_Average * dt

        # Event logic
        if Q_in < Q_DWF_UB:
            event = 0
        elif FD > self.FD_event:
            event = 3
        elif (
            Qsw > self.Qsw_event9
            and self.state["Q_mean"] > self.previous_state["Q_mean"]
        ):
            event = 9
        else:
            event = 0
        self.state["event"] = event

        # Event8_h logic
        event8_h = int(
            (Qsw > self.Qsw_event8)
            and (event < 3)
            and (self.previous_state["proc4_NH4"] > 0.6)
        )
        self.state["event8_h"] = event8_h

        # Event8 logic
        event8 = int(
            (
                (event8_h == 1)
                or (
                    ((t - self.state.get("t_end_event8_h", t)).total_seconds() / 86400)
                    < 0.25
                )
            )
            and (event < 3)
            and ((self.state.get("t_event39", t - t).total_seconds() / 86400) > 0.25)
        )
        self.state["event8"] = event8

        # Event 8 timing
        if (self.previous_state["event8"] == 0) and (event8 == 1):
            self.state["t_start_event8"] = t
        else:
            self.state["t_start_event8"] = self.previous_state.get("t_start_event8", t)

        if (self.previous_state["event8_h"] == 1) and (event8_h == 0):
            self.state["t_end_event8_h"] = t
        else:
            self.state["t_end_event8_h"] = self.previous_state.get("t_end_event8_h", t)

        # NH4 Processes
        proc1_NH4 = self.NH4_av * getattr(self.concentration_dict["NH4"], hour_key)
        self.state["proc1_NH4"] = proc1_NH4

        proc2_NH4 = (self.alpha_NH4 * (Q_DWF_UB / Q_in - 1) + 1) if event == 3 else 1
        self.state["proc2_NH4"] = proc2_NH4

        proc3_NH4 = (self.beta_NH4 * (Q_DWF_UB / Q_in - 1) + 1) if event == 9 else 1
        self.state["proc3_NH4"] = proc3_NH4

        # proc4_NH4 (Ramp)
        self.state["t_proc4"] = t - self.previous_state.get("t_start_proc4", t)
        if (self.previous_state["proc4_NH4"] == 1) and (self.state["proc4_NH4"] < 1):
            self.state["t_start_proc4"] = t
        else:
            self.state["t_start_proc4"] = self.previous_state.get("t_start_proc4", t)

        if event in [3, 9]:
            proc4_NH4_h = 1
        elif self.previous_state["event"] == 3:
            proc4_NH4_h = self.previous_state["proc2_NH4"]
        elif self.previous_state["event"] == 9:
            proc4_NH4_h = self.previous_state["proc3_NH4"]
        else:
            t_proc4 = t - self.previous_state.get("t_start_proc4", t)
            if t_proc4.total_seconds() / 86400 < self.window_proc4_NH4:
                proc4_NH4_h = (
                    self.previous_state["proc4_NH4"] + self.proc4_slope1_NH4 * dt
                )
            else:
                proc4_NH4_h = (
                    self.previous_state["proc4_NH4"] + self.proc4_slope2_NH4 * dt
                )
        self.state["proc4_NH4"] = min(proc4_NH4_h, 1)

        # proc5 NH4 (event8 influence)
        if event8 == 0:
            proc5_1_NH4 = 0
            proc5_2_NH4 = 0
        else:
            if (t - self.previous_state["t_start_proc5_1"]).total_seconds() / 86400 < (
                self.window_proc5_NH4 / 2
            ):
                proc5_1_NH4 = (
                    self.previous_state["proc5_1_NH4"] - self.proc5_slope_NH4 * dt
                )
            elif (
                t - self.previous_state["t_start_proc5_1"]
            ).total_seconds() / 86400 < self.window_proc5_NH4:
                proc5_1_NH4 = (
                    self.previous_state["proc5_1_NH4"] + self.proc5_slope_NH4 * dt
                )
            else:
                proc5_1_NH4 = 0

            if (
                self.previous_state["t_start_proc5_2"]
                <= self.previous_state["t_start_proc5_1"]
            ):
                proc5_2_NH4 = 0
            elif (
                t - self.previous_state["t_start_proc5_2"]
            ).total_seconds() / 86400 < (self.window_proc5_NH4 / 2):
                proc5_2_NH4 = (
                    self.previous_state["proc5_2_NH4"] - self.proc5_slope_NH4 * dt
                )
            elif (
                t - self.previous_state["t_start_proc5_2"]
            ).total_seconds() / 86400 < self.window_proc5_NH4:
                proc5_2_NH4 = (
                    self.previous_state["proc5_2_NH4"] + self.proc5_slope_NH4 * dt
                )
            else:
                proc5_2_NH4 = 0

        if (
            self.previous_state["event8"] == 0
            and self.state["event8"] == 1
            and self.previous_state["t_start_proc5_2"] == self.state["t_start_proc5_2"]
        ):
            self.state["t_start_proc5_1"] = t
        else:
            self.state["t_start_proc5_1"] = self.previous_state.get(
                "t_start_proc5_1", t
            )

        if (
            self.previous_state["event8"] == 0
            and self.state["event8"] == 1
            and (t - self.previous_state["t_start_proc5_1"]).total_seconds() / 86400
            < self.window_proc5_NH4
            and self.previous_state["proc5_2_NH4"] == 0
        ):
            self.state["t_start_proc5_2"] = t
        else:
            self.state["t_start_proc5_2"] = self.previous_state.get(
                "t_start_proc5_2", t
            )

        self.state["proc5_1_NH4"] = proc5_1_NH4
        self.state["proc5_2_NH4"] = proc5_2_NH4
        self.state["proc5_NH4"] = 1 + proc5_1_NH4 + proc5_2_NH4

        # NH4 Final
        self.state["NH4"] = (
            proc1_NH4
            * proc2_NH4
            * proc3_NH4
            * self.state["proc4_NH4"]
            * self.state["proc5_NH4"]
        )

        # CODs Processes
        proc1_CODs = self.CODs_av * getattr(self.concentration_dict["CODs"], hour_key)
        self.state["proc1_CODs"] = proc1_CODs

        proc2_CODs = (self.alpha_CODs * (Q_DWF_UB / Q_in - 1) + 1) if event == 3 else 1
        self.state["proc2_CODs"] = proc2_CODs

        proc3_CODs = (self.beta_CODs * (Q_DWF_UB / Q_in - 1) + 1) if event == 9 else 1
        self.state["proc3_CODs"] = proc3_CODs

        # proc4_CODs (Ramp)
        self.state["t_proc4"] = t - self.previous_state.get("t_start_proc4", t)
        if (self.previous_state["proc4_CODs"] == 1) and (self.state["proc4_CODs"] < 1):
            self.state["t_start_proc4"] = t
        else:
            self.state["t_start_proc4"] = self.previous_state.get("t_start_proc4", t)

        if event in [3, 9]:
            proc4_CODs_h = 1
        elif self.previous_state["event"] == 3:
            proc4_CODs_h = self.previous_state["proc2_CODs"]
        elif self.previous_state["event"] == 9:
            proc4_CODs_h = self.previous_state["proc3_CODs"]
        else:
            t_proc4 = t - self.previous_state.get("t_start_proc4", t)
            if t_proc4.total_seconds() / 86400 < self.window_proc4_CODs:
                proc4_CODs_h = (
                    self.previous_state["proc4_CODs"] + self.proc4_slope1_CODs * dt
                )
            else:
                proc4_CODs_h = (
                    self.previous_state["proc4_CODs"] + self.proc4_slope2_CODs * dt
                )
        self.state["proc4_CODs"] = min(proc4_CODs_h, 1)

        # proc5 CODs (event8 influence)
        if event8 == 0:
            proc5_1_CODs = 0
            proc5_2_CODs = 0
        else:
            if (t - self.previous_state["t_start_proc5_1"]).total_seconds() / 86400 < (
                self.window_proc5_CODs / 2
            ):
                proc5_1_CODs = (
                    self.previous_state["proc5_1_CODs"] - self.proc5_slope_CODs * dt
                )
            elif (
                t - self.previous_state["t_start_proc5_1"]
            ).total_seconds() / 86400 < self.window_proc5_CODs:
                proc5_1_CODs = (
                    self.previous_state["proc5_1_CODs"] + self.proc5_slope_CODs * dt
                )
            else:
                proc5_1_CODs = 0

            if (
                self.previous_state["t_start_proc5_2"]
                <= self.previous_state["t_start_proc5_1"]
            ):
                proc5_2_CODs = 0
            elif (
                t - self.previous_state["t_start_proc5_2"]
            ).total_seconds() / 86400 < (self.window_proc5_CODs / 2):
                proc5_2_CODs = (
                    self.previous_state["proc5_2_CODs"] - self.proc5_slope_CODs * dt
                )
            elif (
                t - self.previous_state["t_start_proc5_2"]
            ).total_seconds() / 86400 < self.window_proc5_CODs:
                proc5_2_CODs = (
                    self.previous_state["proc5_2_CODs"] + self.proc5_slope_CODs * dt
                )
            else:
                proc5_2_CODs = 0

        if (
            self.previous_state["event8"] == 0
            and self.state["event8"] == 1
            and self.previous_state["t_start_proc5_2"] == self.state["t_start_proc5_2"]
        ):
            self.state["t_start_proc5_1"] = t
        else:
            self.state["t_start_proc5_1"] = self.previous_state.get(
                "t_start_proc5_1", t
            )

        if (
            self.previous_state["event8"] == 0
            and self.state["event8"] == 1
            and (t - self.previous_state["t_start_proc5_1"]).total_seconds() / 86400
            < self.window_proc5_CODs
            and self.previous_state["proc5_2_CODs"] == 0
        ):
            self.state["t_start_proc5_2"] = t
        else:
            self.state["t_start_proc5_2"] = self.previous_state.get(
                "t_start_proc5_2", t
            )

        self.state["proc5_1_CODs"] = proc5_1_CODs
        self.state["proc5_2_CODs"] = proc5_2_CODs
        self.state["proc5_CODs"] = 1 + proc5_1_CODs + proc5_2_CODs

        # CODs Final
        self.state["CODs"] = (
            proc1_CODs
            * proc2_CODs
            * proc3_CODs
            * self.state["proc4_CODs"]
            * self.state["proc5_CODs"]
        )

        # PO4 Processes
        proc1_PO4 = self.PO4_av * getattr(self.concentration_dict["PO4"], hour_key)
        self.state["proc1_PO4"] = proc1_PO4

        proc2_PO4 = (self.alpha_PO4 * (Q_DWF_UB / Q_in - 1) + 1) if event == 3 else 1
        self.state["proc2_PO4"] = proc2_PO4

        proc3_PO4 = (self.beta_PO4 * (Q_DWF_UB / Q_in - 1) + 1) if event == 9 else 1
        self.state["proc3_PO4"] = proc3_PO4

        # proc4_PO4 (Ramp)
        self.state["t_proc4"] = t - self.previous_state.get("t_start_proc4", t)
        if (self.previous_state["proc4_PO4"] == 1) and (self.state["proc4_PO4"] < 1):
            self.state["t_start_proc4"] = t
        else:
            self.state["t_start_proc4"] = self.previous_state.get("t_start_proc4", t)

        if event in [3, 9]:
            proc4_PO4_h = 1
        elif self.previous_state["event"] == 3:
            proc4_PO4_h = self.previous_state["proc2_PO4"]
        elif self.previous_state["event"] == 9:
            proc4_PO4_h = self.previous_state["proc3_PO4"]
        else:
            t_proc4 = t - self.previous_state.get("t_start_proc4", t)
            if t_proc4.total_seconds() / 86400 < self.window_proc4_PO4:
                proc4_PO4_h = (
                    self.previous_state["proc4_PO4"] + self.proc4_slope1_PO4 * dt
                )
            else:
                proc4_PO4_h = (
                    self.previous_state["proc4_PO4"] + self.proc4_slope2_PO4 * dt
                )
        self.state["proc4_PO4"] = min(proc4_PO4_h, 1)

        # proc5 PO4 (event8 influence)
        if event8 == 0:
            proc5_1_PO4 = 0
            proc5_2_PO4 = 0
        else:
            if (t - self.previous_state["t_start_proc5_1"]).total_seconds() / 86400 < (
                self.window_proc5_PO4 / 2
            ):
                proc5_1_PO4 = (
                    self.previous_state["proc5_1_PO4"] - self.proc5_slope_PO4 * dt
                )
            elif (
                t - self.previous_state["t_start_proc5_1"]
            ).total_seconds() / 86400 < self.window_proc5_PO4:
                proc5_1_PO4 = (
                    self.previous_state["proc5_1_PO4"] + self.proc5_slope_PO4 * dt
                )
            else:
                proc5_1_PO4 = 0

            if (
                self.previous_state["t_start_proc5_2"]
                <= self.previous_state["t_start_proc5_1"]
            ):
                proc5_2_PO4 = 0
            elif (
                t - self.previous_state["t_start_proc5_2"]
            ).total_seconds() / 86400 < (self.window_proc5_PO4 / 2):
                proc5_2_PO4 = (
                    self.previous_state["proc5_2_PO4"] - self.proc5_slope_PO4 * dt
                )
            elif (
                t - self.previous_state["t_start_proc5_2"]
            ).total_seconds() / 86400 < self.window_proc5_PO4:
                proc5_2_PO4 = (
                    self.previous_state["proc5_2_PO4"] + self.proc5_slope_PO4 * dt
                )
            else:
                proc5_2_PO4 = 0

        if (
            self.previous_state["event8"] == 0
            and self.state["event8"] == 1
            and self.previous_state["t_start_proc5_2"] == self.state["t_start_proc5_2"]
        ):
            self.state["t_start_proc5_1"] = t
        else:
            self.state["t_start_proc5_1"] = self.previous_state.get(
                "t_start_proc5_1", t
            )

        if (
            self.previous_state["event8"] == 0
            and self.state["event8"] == 1
            and (t - self.previous_state["t_start_proc5_1"]).total_seconds() / 86400
            < self.window_proc5_PO4
            and self.previous_state["proc5_2_PO4"] == 0
        ):
            self.state["t_start_proc5_2"] = t
        else:
            self.state["t_start_proc5_2"] = self.previous_state.get(
                "t_start_proc5_2", t
            )

        self.state["proc5_1_PO4"] = proc5_1_PO4
        self.state["proc5_2_PO4"] = proc5_2_PO4
        self.state["proc5_PO4"] = 1 + proc5_1_PO4 + proc5_2_PO4

        # PO4 Final
        self.state["PO4"] = (
            proc1_PO4
            * proc2_PO4
            * proc3_PO4
            * self.state["proc4_PO4"]
            * self.state["proc5_PO4"]
        )

        # COD Processes
        # proc1_COD
        proc1_COD = self.COD_av * getattr(self.concentration_dict["COD"], hour_key)
        self.state["proc1_COD"] = proc1_COD

        # proc2_COD
        if self.state["event"] == 3:
            proc2_COD = (
                self.alpha_COD * (self.state["Q_DWF_UB"] / self.state["Q_in"] - 1) + 1
            )
        else:
            proc2_COD = 1
        self.state["proc2_COD"] = proc2_COD

        # proc3_COD
        if self.state["event"] == 9:
            proc3_COD = (
                self.beta_COD * (self.state["Q_DWF_UB"] / self.state["Q_in"] - 1) + 1
            )
        else:
            proc3_COD = 1
        self.state["proc3_COD"] = proc3_COD

        # proc4_COD (Ramp)
        if self.state["event"] in [3, 9]:
            proc4_COD_h = 1
        elif self.previous_state["event"] == 3:
            proc4_COD_h = self.previous_state["proc2_COD"]
        elif self.previous_state["event"] == 9:
            proc4_COD_h = self.previous_state["proc3_COD"]
        else:
            t_proc4 = t - self.previous_state.get("t_start_proc4", t)
            if t_proc4.total_seconds() / 86400 < self.window_proc4_COD:
                proc4_COD_h = (
                    self.previous_state["proc4_COD"] + self.proc4_slope1_COD * dt
                )
            else:
                proc4_COD_h = (
                    self.previous_state["proc4_COD"] + self.proc4_slope2_COD * dt
                )
        self.state["proc4_COD"] = min(proc4_COD_h, 1)

        # proc6_COD_h
        if self.state["event"] in [3, 9] and self.previous_state["event"] < 3:
            if (
                self.state.get("t_event39", t - t).total_seconds() / 86400
                < self.proc6_t1_COD
            ):
                proc6_COD_h = 0
            elif (
                self.state.get("t_event39", t - t).total_seconds() / 86400
                < self.proc6_t2_COD
            ):
                proc6_COD_h = self.peak_COD_low
            else:
                proc6_COD_h = self.peak_COD_high
        elif self.state["event"] in [3, 9] and self.state["Q_in"] > self.Q_proc6:
            proc6_COD_h = (
                self.previous_state.get("proc6_COD_h", t).hour
                - self.proc6_slope1_COD * dt
            )
        else:
            proc6_COD_h = (
                self.previous_state.get("proc6_COD_h", t).hour
                - self.proc6_slope2_COD * dt
            )
        self.state["proc6_COD"] = max(proc6_COD_h, 0)

        if (
            self.previous_state["event"] == 3 or self.previous_state["event"]
        ) == 9 and self.state["event"] < 3:
            self.state["t_end_event39"] = t
        else:
            self.state["t_end_event39"] = self.previous_state.get("t_end_event39", t)

        self.state["t_event39"] = t - self.state.get("t_end_event39", t)

        # proc7_COD_h
        if (
            self.state["event"] in [3, 9]
            or self.state.get("t_event39", t - t).total_seconds() / 86400 < 1.5
        ):
            self.state["proc7_COD_h"] = 0
        elif self.state["event8"] == 1:
            if (
                t - self.state.get("t_start_event8", t)
            ).total_seconds() / 86400 < 0.1333:
                self.state["proc7_COD_h"] = (
                    self.previous_state["proc7_COD_h"] + self.proc7_slope1_COD * dt
                )
            else:
                self.state["proc7_COD_h"] = (
                    self.previous_state["proc7_COD_h"] - self.proc7_slope2_COD * dt
                )
        else:
            self.state["proc7_COD_h"] = 0
        self.state["proc7_COD"] = max(self.state["proc7_COD_h"], 0)

        # COD Final Calculation
        self.state["COD"] = (
            self.state["proc1_COD"]
            * self.state["proc2_COD"]
            * self.state["proc3_COD"]
            * self.state["proc4_COD"]
            + self.state["proc6_COD"]
            + self.state["proc7_COD"]
        )

        # TSS Processes
        proc1_TSS = self.TSS_av * getattr(self.concentration_dict["TSS"], hour_key)
        self.state["proc1_TSS"] = proc1_TSS

        # proc2_TSS
        if self.state["event"] == 3:
            proc2_TSS = (
                self.alpha_TSS * (self.state["Q_DWF_UB"] / self.state["Q_in"] - 1) + 1
            )
        else:
            proc2_TSS = 1
        self.state["proc2_TSS"] = proc2_TSS

        # proc3_TSS
        if self.state["event"] == 9:
            proc3_TSS = (
                self.beta_TSS * (self.state["Q_DWF_UB"] / self.state["Q_in"] - 1) + 1
            )
        else:
            proc3_TSS = 1
        self.state["proc3_TSS"] = proc3_TSS

        # proc4_TSS (Ramp)
        if self.state["event"] in [3, 9]:
            proc4_TSS_h = 1
        elif self.previous_state["event"] == 3:
            proc4_TSS_h = self.previous_state["proc2_TSS"]
        elif self.previous_state["event"] == 9:
            proc4_TSS_h = self.previous_state["proc3_TSS"]
        else:
            t_proc4 = t - self.previous_state.get("t_start_proc4", t)
            if t_proc4.total_seconds() / 86400 < self.window_proc4_TSS:
                proc4_TSS_h = (
                    self.previous_state["proc4_TSS"] + self.proc4_slope1_TSS * dt
                )
            else:
                proc4_TSS_h = (
                    self.previous_state["proc4_TSS"] + self.proc4_slope2_TSS * dt
                )
        self.state["proc4_TSS"] = min(proc4_TSS_h, 1)

        # proc6_TSS_h
        if self.state["event"] in [3, 9] and self.previous_state["event"] < 3:
            if (
                self.state.get("t_event39", t - t).total_seconds() / 86400
                < self.proc6_t1_TSS
            ):
                proc6_TSS_h = 0
            elif (
                self.state.get("t_event39", t - t).total_seconds() / 86400
                < self.proc6_t2_TSS
            ):
                proc6_TSS_h = self.peak_TSS_low
            else:
                proc6_TSS_h = self.peak_TSS_high
        elif self.state["event"] in [3, 9] and self.state["Q_in"] > self.Q_proc6:
            proc6_TSS_h = (
                self.previous_state.get("proc6_TSS_h", t).hour
                - self.proc6_slope1_TSS * dt
            )
        else:
            proc6_TSS_h = (
                self.previous_state.get("proc6_TSS_h", t).hour
                - self.proc6_slope2_TSS * dt
            )
        self.state["proc6_TSS"] = max(proc6_TSS_h, 0)

        if (
            self.previous_state["event"] == 3 or self.previous_state["event"]
        ) == 9 and self.state["event"] < 3:
            self.state["t_end_event39"] = t
        else:
            self.state["t_end_event39"] = self.previous_state.get("t_end_event39", t)

        self.state["t_event39"] = t - self.state.get("t_end_event39", t)

        # proc7_TSS_h
        if (
            self.state["event"] in [3, 9]
            or self.state.get("t_event39", t - t).total_seconds() / 86400 < 1.5
        ):
            self.state["proc7_TSS_h"] = 0
        elif self.state["event8"] == 1:
            if (
                t - self.state.get("t_start_event8", t)
            ).total_seconds() / 86400 < 0.1333:
                self.state["proc7_TSS_h"] = (
                    self.previous_state["proc7_TSS_h"] + self.proc7_slope1_TSS * dt
                )
            else:
                self.state["proc7_TSS_h"] = (
                    self.previous_state["proc7_TSS_h"] - self.proc7_slope2_TSS * dt
                )
        else:
            self.state["proc7_TSS_h"] = 0
        self.state["proc7_TSS"] = max(self.state["proc7_TSS_h"], 0)

        # TSS Final Calculation
        self.state["TSS"] = (
            self.state["proc1_TSS"]
            * self.state["proc2_TSS"]
            * self.state["proc3_TSS"]
            * self.state["proc4_TSS"]
            + self.state["proc6_TSS"]
            + self.state["proc7_TSS"]
        )

        self.log["H2O_sew"].append(
            -inflow_H2O_sew * 1_000_000
        )  # m3/d to g/d conversion
        self.log["COD_part"].append(
            -(self.state["COD"] - self.state["CODs"]) * self.state["Q_in"]
        )
        self.log["COD_sol"].append(-self.state["CODs"] * self.state["Q_in"])
        self.log["X_TSS_sew"].append(-self.state["TSS"] * self.state["Q_in"])
        self.log["NH4_sew"].append(-self.state["NH4"] * self.state["Q_in"])
        self.log["PO4_sew"].append(-self.state["PO4"] * self.state["Q_in"])
        self.index.append(t)

        # Update time
        self.state["t"] = t
        self.previous_state = self.state.copy()
