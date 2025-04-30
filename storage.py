class Storage:
    def __init__(self, max_volume):
        self.V_max = max_volume  # m3
        self.stored_volume = 0

    def FD(self):
        return self.stored_volume / self.V_max

    def update_stored_volume(self, volume):
        if volume < 0:
            self.stored_volume = 0
        else:
            self.stored_volume = volume


class RZ_storage(Storage):
    def __init__(
        self,
        volume,
        pipes=[
            "Con_103",
            "Con_104",
            "Con_105",
            "Con_106",
            "Con_107",
            "Con_108",
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
            "Con_152",
            "Con_153",
            "Con_154",
            "Con_155",
            "Con_156",
        ],
    ):
        super().__init__(volume)  # Initialize the parent class with V_max
        self.pipes = pipes

    def get_volume(self, links):
        pipe_volumes = [links[pipe].volume for pipe in self.pipes]
        return sum(pipe_volumes)


class ConcentrationStorage:
    def __init__(self):
        self.V = 0
        self.storage_concentrations = {
            "COD_part": 0,
            "COD_sol": 0,
            "X_TSS_sew": 0,
            "NH4_sew": 0,
            "PO4_sew": 0,
        }

    def update_in(self, Q, pollutant_load, timestep=300):
        V_in = Q * timestep

        for k, i in self.storage_concentrations.items():
            self.storage_concentrations[k] = (
                self.V * i + pollutant_load[k] * (300 / 86400)
            ) / (self.V + V_in)
        self.V += V_in

    def update_out(self, Q, timestep=300):
        V_out = Q * timestep  # CMS to CM
        conc_out = {
            k: v * Q * 86400 for k, v in self.storage_concentrations.items()
        }  # g/m3 to g/d
        conc_out["H2O_sew"] = Q * 86400
        self.V -= V_out
        return conc_out
