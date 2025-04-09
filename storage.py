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
    def __init__(self):
        super().__init__(21407.57)  # Initialize the parent class with V_max
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
