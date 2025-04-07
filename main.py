from simulation import Simulation
from postprocess import PostProcess
import datetime as dt

# MODEL_NAME = "model_jip_geen_regen"
# MODEL_NAME = "model_jip_WEST_regen"
MODEL_NAME = "model_jip"
SUFFIX = "RTC_result_1"


def main():
    simulation = Simulation(
        model_path=rf"data\SWMM\{MODEL_NAME}.inp",
        step_size=300,
        # Extended summer period:
        # report_start=dt.datetime(year=2024, month=4, day=15),
        # start_time=dt.datetime(year=2024, month=4, day=15),
        # end_time=dt.datetime(year=2024, month=10, day=16),
        report_start=dt.datetime(year=2024, month=7, day=1),
        start_time=dt.datetime(year=2024, month=7, day=1),
        end_time=dt.datetime(year=2024, month=8, day=1),
        virtual_pump_max=10,
    )
    simulation.start_simulation()

    postprocess = PostProcess(model_name=MODEL_NAME)
    # postprocess.create_outfall_txt(suffix=SUFFIX)

    # postprocess.plot_outfalls(save=False, plot_rain=True, suffix=SUFFIX)
    postprocess.plot_pumps(
        save=True, plot_rain=True, suffix=SUFFIX, target_setting=True
    )
    # TODO postprocess.plot_storages()


if __name__ == "__main__":
    main()
