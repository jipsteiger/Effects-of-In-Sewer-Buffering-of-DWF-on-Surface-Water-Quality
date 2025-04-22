from concentration_curves import concentration_dict


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
    def __init__(self, volume):
        super().__init__(volume)  # Initialize the parent class with V_max
        self.pipes = [
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
        ]

    def get_volume(self, links):
        pipe_volumes = [links[pipe].volume for pipe in self.pipes]
        return sum(pipe_volumes)


class ConcentrationStorage:
    def __init__(self):
        self.V = 0
        self.storage_concentrations = {
            "COD": 0,
            "CODs": 0,
            "TSS": 0,
            "NH4": 0,
            "PO4": 0,
        }

    def update_in(self, Q, h, timestep):
        V_in = Q * timestep

        for k, i in self.storage_concentrations.items():
            hour_key = f"H_{int(h)}"  # ensure h is int or convert it
            conc_in = getattr(concentration_dict[k], hour_key)
            self.storage_concentrations[k] = (self.V * i + V_in * conc_in) / (
                self.V + V_in
            )
        self.V += V_in

    def update_out(self, Q, timestep):
        V_out = Q * timestep
        conc_out = self.storage_concentrations
        self.V -= V_out
        return conc_out
