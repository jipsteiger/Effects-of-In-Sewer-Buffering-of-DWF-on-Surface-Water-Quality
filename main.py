from simulation import Simulation
from postprocess import PostProcess
import datetime as dt


MODEL_NAME = "model_jip"


def main():
    simulation = Simulation(
        model_path=rf"data\SWMM\{MODEL_NAME}.inp",
        step_size=300,
        report_start=dt.datetime(year=2024, month=9, day=1),
        start_time=dt.datetime(year=2024, month=9, day=1),
        end_time=dt.datetime(year=2024, month=9, day=30),
        virtual_pump_max=50,
    )
    simulation.start_simulation()

    postprocess = PostProcess(model_name=MODEL_NAME)
    postprocess.create_outfall_csv()

    postprocess.plot_outfalls(save=True, plot_rain=True)
    postprocess.plot_pumps(save=True, plot_rain=True)
    # TODO postprocess.plot_storages()


if __name__ == "__main__":
    main()
