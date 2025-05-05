from simulation import Simulation
from realtimecontrol import RealTimeControl
from postprocess import PostProcess
import datetime as dt

# MODEL_NAME = "model_jip_geen_regen"
# MODEL_NAME = "model_jip_WEST_regen"
MODEL_NAME = "model_jip"
# MODEL_NAME = "model_jip_no_rtc"

# MODEL_NAME = "model_jip_with_pump_curve"

SUFFIX = "RTC"


def main():
    simulation = RealTimeControl(
        model_path=rf"data\SWMM\{MODEL_NAME}.inp",
        step_size=300,
        # Extended summer period:
        # report_start=dt.datetime(year=2024, month=4, day=15),
        # start_time=dt.datetime(year=2024, month=4, day=15),
        # end_time=dt.datetime(year=2024, month=10, day=16),
        report_start=dt.datetime(year=2024, month=7, day=1),
        start_time=dt.datetime(year=2024, month=7, day=1),
        end_time=dt.datetime(year=2024, month=7, day=31),
        virtual_pump_max=10,
        constant_outflow=False,
    )
    simulation.start_simulation()

    postprocess = PostProcess(model_name=MODEL_NAME)
    # postprocess.create_outfall_txt(suffix=SUFFIX)
    postprocess.create_outfall_txt_concentrate(suffix=SUFFIX, specific_version="RTC")

    # postprocess.plot_outfalls(save=False, plot_rain=True, suffix=SUFFIX)
    # postprocess.plot_pumps(
    #     save=True, plot_rain=True, suffix=SUFFIX, target_setting=True, storage=True
    # )


if __name__ == "__main__":
    main()
