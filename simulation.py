import pyswmm as ps
import datetime as dt
import pandas as pd
from pathlib import Path
from storage import ConcentrationStorage
from emprical_sewer_wq import EmpericalSewerWQ
from storage import Storage, RZ_storage
from data.concentration_curves import concentration_dict_ES, concentration_dict_RZ


class Simulation:
    def __init__(
        self,
        model_path: str,
        step_size: int,
        report_start: dt.datetime,
        start_time: dt.datetime,
        end_time: dt.datetime,
        virtual_pump_max=10,
        constant_outflow=False,
    ):
        self.model_path = model_path
        self.step_size = step_size
        self.report_start = report_start
        self.start_time = start_time
        self.end_time = end_time

        self.virtual_pump_max = virtual_pump_max
        self.constant_outflow = constant_outflow

        # For EmpericalSewerWQ the Filling Degree of the storage, upstream of the WQ model
        # is required.
        self.ES_storage_FD = Storage(165000)
        self.RZ_storage_FD = RZ_storage(
            35721,
            pipes=[
                "Con_161",
                "Con_103",
                "Con_104",
                "Con_105",
                "Con_106",
                "Con_162",
                "Con_147",
                "Con_148",
                "Con_158",
                "Con_107",
                "Con_108",
                "Con_110",
                "Con_157",
                "Con_111",
                "Con_112",
                "Con_113",
                "Con_114",
                "Con_115",
                "Con_116",
                "Con_117",
                "Con_118",
                "Con_119",
                "Con_120",
                "Con_121",
                "Con_122",
                "Con_123",
                "Con_141",
                "Con_142",
                "Con_143",
                "Con_144",
                "Con_152",
                "Con_153",
                "Con_154",
                "Con_155",
                "Con_156",
                "Con_159",
                "Con_160",
                "Con_109",
            ],
        )

        self.ESConcentrationStorage = ConcentrationStorage()
        self.RZConcentrationStorage = ConcentrationStorage()
        self.ESconcentration_df = pd.DataFrame(
            columns=["COD", "CODs", "TSS", "NH4", "PO4"]
        )
        self.RZconcentration_df = pd.DataFrame(
            columns=["COD", "CODs", "TSS", "NH4", "PO4"]
        )

        self.precipitation_forecast = get_precipitation()

    def start_simulation(self):
        with ps.Simulation(
            self.model_path,
        ) as self.sim:
            self.set_simulation_settings()
            self.get_links_and_nodes()
            self.init_virtual_storage()
            self.init_emperical_sewer_WQ()

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

    def init_emperical_sewer_WQ(self):
        self.WQ_RZ = EmpericalSewerWQ(
            concentration_dict=concentration_dict_RZ,
            COD_av=573,
            CODs_av=206,
            NH4_av=44,
            PO4_av=7.1,
            TSS_av=203,
            Q_95_av=58800,
            alpha_CODs=0.8,
            alpha_NH4=0.8,
            alpha_PO4=0.8,
            alpha_TSS=0.8,
            beta_CODs=0.8,
            beta_NH4=0.8,
            beta_PO4=0.8,
            beta_TSS=0.8,
            proc4_slope1_CODs=0.288,
            proc4_slope1_NH4=0.288,
            proc4_slope1_PO4=0.288,
            proc4_slope2_CODs=0.576,
            proc4_slope2_NH4=0.576,
            proc4_slope2_PO4=0.576,
            Q_proc6=120000,
            proc6_slope2_COD=864,
            proc6_slope2_TSS=864,
            proc6_t1_COD=3,
            proc6_t1_TSS=3,
            proc6_t2_COD=12,
            proc6_t2_TSS=12,
            proc7_slope1_COD=23040,
            proc7_slope1_TSS=23040,
            proc7_slope2_COD=5760,
            proc4_slope2_TSS=5760,
        )
        self.WQ_ES = EmpericalSewerWQ(
            concentration_dict=concentration_dict_ES,
            COD_av=546,
            CODs_av=158,
            NH4_av=44,
            PO4_av=7.1,
            TSS_av=255,
            Q_95_av=70800,
            alpha_CODs=0.8,
            alpha_NH4=0.8,
            alpha_PO4=0.8,
            alpha_TSS=0.8,
            beta_CODs=0.8,
            beta_NH4=0.8,
            beta_PO4=0.8,
            beta_TSS=0.8,
        )

    def simulation_steps(self):
        for step in self.sim:
            self.handle_virtual_storage()
            self.handle_c_119_flows()
            self.handle_geldrop_out_flows()

            if self.constant_outflow and (self.model_path == "data\SWMM\model_jip.inp"):
                self.do_constant_outflow()
            if self.constant_outflow and not (
                self.model_path == "data\SWMM\model_jip.inp"
            ):
                print(
                    f"Constant outflow enabled, but wrong model is used, therefor regular outflow is done."
                )
            self.set_storage_for_FD()
            self.update_WQ()

            self.concentrations()
        self.WQ_ES.write_output_log("ES")
        self.WQ_RZ.write_output_log("RZ")
        self.save_concentrations()

    def do_constant_outflow(self):
        self.links["P_eindhoven_out"].target_setting = 0.663 / 3.888

        self.links["P_riool_zuid_out"].target_setting = 4.7222 / 0.5218

    def update_WQ(self):
        # Below inflows are used for testing
        # RZ_in = self.links["P_riool_zuid_out"].flow * 3600 * 24
        # RZ_in = self.nodes["out_RZ"].total_inflow * 3600 * 24
        RZ_in = (
            (self.nodes["Nod_112"].total_inflow + self.nodes["Nod_104"].total_inflow)
            * 3600
            * 24
        )
        RZ_FD = self.RZ_storage_FD.FD()
        self.WQ_RZ.update(self.sim.current_time, RZ_in, RZ_FD)

        # Below inflows are used for testing
        # ES_in = self.links["P_eindhoven_out"].flow * 3600 * 24
        # ES_in = self.nodes["out_ES"].total_inflow * 3600 * 24
        ES_in = self.nodes["pipe_ES"].total_inflow * 3600 * 24
        ES_FD = self.ES_storage_FD.FD()
        self.WQ_ES.update(self.sim.current_time, ES_in, ES_FD)

        self.RZ_inflow = self.WQ_RZ.get_latest_log()  # Returns pollutant flow in g/d
        self.ES_inflow = self.WQ_ES.get_latest_log()

    def concentrations(self):
        if (
            type(self) is not Simulation
        ):  # Is required to mimic better the concentration levels at the outflow if no RTC is used.
            self.ESConcentrationStorage.update_in(
                self.nodes["pipe_ES"].total_inflow, self.ES_inflow
            )
            ESconcentration_out = self.ESConcentrationStorage.update_out(
                self.links["P_eindhoven_out"].flow, self.ES_storage_FD.FD()
            )
        else:
            ESconcentration_out = self.ES_inflow
        ESrow_df = pd.DataFrame([ESconcentration_out], index=[self.sim.current_time])

        if self.ESconcentration_df.empty:
            self.ESconcentration_df = ESrow_df.copy()
        else:
            self.ESconcentration_df = pd.concat([self.ESconcentration_df, ESrow_df])

        if type(self) is not Simulation:
            RZ_in = (
                self.nodes["Nod_112"].total_inflow + self.nodes["Nod_104"].total_inflow
            )
            self.RZConcentrationStorage.update_in(RZ_in, self.RZ_inflow)
            RZconcentration_out = self.RZConcentrationStorage.update_out(
                self.links["P_riool_zuid_out"].flow, self.RZ_storage_FD.FD()
            )
        else:
            RZconcentration_out = self.ES_inflow
        RZrow_df = pd.DataFrame([RZconcentration_out], index=[self.sim.current_time])
        if self.RZconcentration_df.empty:
            self.RZconcentration_df = RZrow_df.copy()
        else:
            self.RZconcentration_df = pd.concat([self.RZconcentration_df, RZrow_df])

    def save_concentrations(self, name="NO_RTC"):
        model_name = Path(self.model_path).stem
        self.ESconcentration_df.to_csv(
            f"output_effluent/{model_name}_ES_effluent_conc_{name}.csv"
        )
        self.RZconcentration_df.to_csv(
            f"output_effluent/{model_name}_RZ_effluent_conc_{name}.csv"
        )

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

    def set_storage_for_FD(self):
        self.ES_storage_FD.update_stored_volume(self.nodes["pipe_ES"].volume)
        self.RZ_storage_FD.update_stored_volume(
            self.RZ_storage_FD.get_volume(self.links)
        )


def get_precipitation():
    precipitation = pd.read_csv(
        rf"data\precipitation\csv_selected_area_euradclim\2024_5_min_precipitation_data.csv",
        index_col=0,
        parse_dates=True,
    )
    return precipitation.resample("h").sum()


##################################################################################################
if __name__ == "__main__":
    MODEL_NAME = "model_jip_with_pump_curve"
    suffix = "Reg. sim"
    simulation = Simulation(
        model_path=rf"data\SWMM\{MODEL_NAME}.inp",
        step_size=300,
        report_start=dt.datetime(year=2024, month=7, day=1),
        start_time=dt.datetime(year=2024, month=7, day=1),
        end_time=dt.datetime(year=2024, month=8, day=1),
        virtual_pump_max=10,
    )
    simulation.start_simulation()

    from postprocess import PostProcess

    postprocess = PostProcess(model_name=MODEL_NAME)
    postprocess.plot_pumps(
        save=False, plot_rain=True, target_setting=True, suffix=suffix, storage=True
    )
