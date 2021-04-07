
from src.training.models import SupernetModel
import copy

class HistoricalSupernetMap():
    def __init__(self):
        self.map = {}


    def initialize_map(self,supernet,historical_marker):
        for i in range(historical_marker.blueprint_marks):
            supernet_copy = copy.deepcopy(supernet)
            self.map[i] = {}

            if i == 0:
                self.map[i]["C_in"] = 3
                self.map[i]["C_out"] = 16
            else:
                self.map[i]["C_in"] = self.map[i-1]["C_out"]
                self.map[i]["C_out"] =  self.map[i-1]["C_out"]  * 2
            supernet_copy.id_generator()
            supernet_copy.update_channels_count(self.map[i]["C_in"],self.map[i]["C_out"])
            self.map[i]["supernet_model"] = SupernetModel(supernet_copy)


    def update_map(self, mark, supernet, predecessor):
        C_in = self.map[predecessor]["C_out"]
        self.map[mark] = {}
        self.map[mark]["C_in"] = C_in
        self.map[mark]["C_out"] =  C_in*2
        supernet_copy = copy.deepcopy(supernet)
        supernet_copy.id_generator()
        supernet_copy.update_channels_count(C_in,C_in*2)
        self.map[mark]["supernet_model"] = SupernetModel(supernet_copy)

