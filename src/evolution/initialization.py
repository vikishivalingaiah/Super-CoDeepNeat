from src.encoding.component import Component
from src.encoding.module import Module
from src.encoding.blueprint import Blueprint
from src.encoding.supernet import Supernet
from src.encoding.layer import Layer
from src.encoding import DAGEncoder
from src.training.models import SupernetModel
from src.evolution.historical_supernets import HistoricalSupernetMap
from random import choice, sample, randint
import networkx as nx
import numpy as np
import copy
import logging

class Initialization:
    def __init__(self, layers, additional_layers=None, sequential_components=None, random_complements=False, supernet_layers = None):
        self.layers = layers
        self.additional_layers = additional_layers
        self.sequential_components = sequential_components
        self.random_complements = random_complements
        self.supernet_layers = supernet_layers



    def create_supernet_layers(self):
        if self.supernet_layers is not None:
            layers = []
            for layer in self.supernet_layers:

                possible_hyperparameters = []
                hyperparameter_order = []
                for hyperparameter_name, values in layer["tf.keras.hyperparameters_table"].items():
                    possible_hyperparameters.append(values["values"])
                    hyperparameter_order.append(hyperparameter_name)

                unique_hyperparameter_combination = self.recursive_fucntion(possible_hyperparameters)
                for unique in unique_hyperparameter_combination:
                    hyperparameters = {}
                    for hyperparameter_key,value in zip(hyperparameter_order,unique):
                        hyperparameters[hyperparameter_key] = value
                    full_layer = Layer(layer["tf.keras.layer"], layer["tf.keras.hyperparameters_table"])
                    full_layer.id_generator()
                    full_layer.keras_hyperparameters = hyperparameters
                    layers.append(full_layer)

            return layers
        else:
            raise FileNotFoundError("Supernet config not found")


    def initialize_supernet_components(self):
        components = []
        layers = self.create_supernet_layers()
        for layer in layers:
            node = [(layer.id, {"layer": layer})]
            edge = []
            component = Component(DAGEncoder.build_dag(node))
            component.id_generator()
            components.append(component)
        return components


    def initialize_supernet_from_components(self, population_size, minimal_supernet, components=None):


        supernets = []

        def id_generator():
            unique_number = "".join(sample([str(i) for i in range(0, 9)], 6))
            id = "node" + "-" + unique_number
            return id
        for i in range(population_size):
            graph_size = choice(minimal_supernet)

            nodes = []
            edges = []

            for j in range(graph_size):
                components = self.initialize_supernet_components()
                activations = [0] * len(components)
                print(activations)
                pos = randint(0, len(activations) - 1)
                activations[pos] = 1
                print(activations)
                nodes.append((id_generator(), {"component_list": components, "activations" : activations }))

            for i in range(len(nodes)):
                u = nodes[i]
                for v in nodes[i+1:]:
                    edges.append((u[0],v[0]))
            g = DAGEncoder.build_dag(nodes, edges)
            print(g.nodes)
            print(g.edges)
            supernet = Supernet(g)
            supernet.id_generator()
            supernets.append(supernet)
        return supernets

    def recursive_fucntion(self,list_1):
        if len(list_1) != 1:
            a = []
            for i in list_1[0]:
                for j in self.recursive_fucntion(list_1[1:]):
                    if type(i) is list and type(j) is list:
                        a.append(i + j)
                    elif type(i) is list:
                        a.append(i + [j])
                    elif type(j) is list:
                        a.append([i] + j)
                    else:
                        a.append([i, j])
        else:
            a = list_1[-1]
        return a


    def initialize_modules_from_supernet(self, supernet, population_size, historical_marker):

        def random_bit_flip(bitstring):
            if bitstring == None:
                raise ValueError("Illegal value for bitstring")
            if len(bitstring) < 2:
                logging.debug("single component case, no bitflip required")
                return
            for i in range(len(bitstring)):
                if bitstring[i] == 1:
                    bitstring[i] = 0
                    break
            random_bit = randint(0, len(bitstring) - 1)
            bitstring[random_bit] = 1
            return bitstring

        modules = []
        supernet_graph = supernet.graph
        for i in range(population_size):
            selected = []
            supernet_nodes = list(nx.algorithms.topological_sort(supernet_graph))

            if len(supernet_nodes) > 1:
                current_node = supernet_nodes[0]

                while (supernet_graph.out_degree(current_node) != 0):

                    selected_edge = choice(list(supernet_graph.out_edges(current_node)))
                    selected.append(selected_edge)
                    current_node = selected_edge[1]
                module_graph = copy.deepcopy(supernet_graph.edge_subgraph(selected))
            else:
                module_graph = copy.deepcopy(supernet_graph)

            for node in nx.algorithms.topological_sort(module_graph):
                activations = module_graph.nodes[node]["activations"]
                activations = random_bit_flip(activations)
                module_graph.nodes[node]["activations"] = activations

            module = Module(module_graph, supernet,historical_marker.mark_module())
            module.id_generator()
            modules.append(module)
        return modules

    def initialize_blueprints(self,population_size, minimal_blueprints, modules, historical_marker, supernet, supernets_map):
        blueprints = []
        chosen_sizes = []
        for i in range(population_size):
            graph_size = choice(minimal_blueprints)
            chosen_sizes.append(graph_size)
            marks = range(graph_size)
            nodes = []
            for j in range(graph_size):
                random_module = choice(modules)
                nodes.append((random_module.id + "_" + str(j) , {"historical_mark": marks[j], "module": random_module}))
            edges = [(u[0], v[0]) for u, v in zip(nodes, nodes[1:])]
            g = DAGEncoder.build_dag(nodes, edges)
            mark = historical_marker.mark_blueprint()
            blueprint = Blueprint(g, mark)
            blueprint.id_generator()
            blueprints.append(blueprint)


        historical_marker.blueprint_marks = np.max(np.array(chosen_sizes))
        supernets_map.initialize_map(supernet, historical_marker)


        return blueprints






