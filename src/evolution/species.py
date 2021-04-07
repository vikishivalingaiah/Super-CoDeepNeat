import numpy as np

class Species:

    def __init__(self, id, members, species_type, name=None):
        self.id = id
        self.members = members
        self.species_type = species_type
        self.name = name
        self.species_fitness = [99,0]

    def update_fitness(self):
        if len(self.members) > 0:
            members_fitness = [member.fitness_values for member in self.members]
            avg_loss = np.array(members_fitness)[:,0].mean()
            avg_accuracy = np.array(members_fitness)[:,1].mean()
            self.species_fitness = [avg_loss, avg_accuracy]

    def fitness(self):
        return self.species_fitness[1]




