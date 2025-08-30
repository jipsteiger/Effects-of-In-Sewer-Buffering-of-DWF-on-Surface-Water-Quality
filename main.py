from simulation import Simulation
from realtimecontrol import RealTimeControl
from postprocess import PostProcess
import datetime as dt

MODEL_NAME = "model_jip"

SUFFIX = "RTC"


def main():
    simulation = RealTimeControl(
        model_path=rf"data\SWMM\{MODEL_NAME}.inp",
        step_size=300,
        # Extended summer period:
        report_start=dt.datetime(year=2024, month=4, day=15),
        start_time=dt.datetime(year=2024, month=4, day=15),
        end_time=dt.datetime(year=2024, month=10, day=16),
        virtual_pump_max=10,
        constant_outflow=False,
        use_ensemble_forecast=False,
        ES_threshold=1,
        RZ_threshold=3,
        certainty_threshold=0.9,
    )
    simulation.start_simulation()

    postprocess = PostProcess(model_name=MODEL_NAME)
    # Create txt files to use as input for WWTP sim
    postprocess.create_outfall_txt_concentrate(suffix=SUFFIX, specific_version="RTC")


if __name__ == "__main__":
    main()
