import networkx as nx
from random import sample, randrange, choice
import matplotlib.pyplot as plt
from src.encoding.DAGEncoder import build_dag
import numpy as np
from random import randint

class Blueprint:
    def __init__(self, graph, mark, name = None, identifier = None):
        self.name = name
        self.id = identifier
        self.mark = mark
        self.fitness_values = [99,0]
        self.species = -1
        if (nx.is_directed_acyclic_graph(graph)):
            self.graph = graph
            self.module_count = len(list(graph.nodes))
        else:
            print("The given graph is DAG: %s" % nx.is_directed_acyclic_graph(graph))


    def id_generator(self):
        unique_number = "".join(sample([str(i) for i in range(0, 9)], 6))
        self.id = "blueprint" + "-" + unique_number
        return unique_number

    def add_connection(self):
        nodes = list(nx.algorithms.topological_sort(self.graph))
        if len(nodes) == 1:
            return
        count = 0
        while(True):
            print("stuck in while loop, add_connection_blueprint")
            n1 = randrange(0, len(nodes) - 1)
            n2 = randrange(n1, len(nodes))
            if  nodes[n1] != nodes[n2] and ((nodes[n1], nodes[n2]) not in self.graph.edges) and self.graph.out_degree(nodes[n1]) != 0:
                self.graph.add_edge(nodes[n1], nodes[n2])
                break
            elif count == 50:
                break
            count = count + 1

    def switch_module(self, modules):
        nodes = list(nx.algorithms.topological_sort(self.graph))
        selected_node = choice(nodes)
        selected_module = choice(modules)
        self.graph.nodes[selected_node]["module"] = selected_module


    def add_node_in_edge(self, modules, historical_marker, supernet, supernets_map):
        nodes = list(nx.algorithms.topological_sort(self.graph))
        if len(nodes) == 1:
            return
        while (True):
            print("stuck in while loop, add_connection_node_in edge")
            n1 = randrange(0, len(nodes) - 1)
            n2 = randrange(n1, len(nodes))
            if ((nodes[n1], nodes[n2]) in self.graph.edges) and self.graph.out_degree(nodes[n1]) != 0 and nodes[n1] != nodes[n2]:
                break


        new_module = choice(modules)
        self.module_id_generator()
        node = new_module.id + "_" + str(self.module_count)
        self.graph.remove_edge(nodes[n1], nodes[n2])
        predecessor_mark = self.graph.nodes[nodes[n1]]["historical_mark"]
        mark = historical_marker.mark_blueprint_genes()
        supernets_map.update_map(mark, supernet,predecessor_mark)
        self.graph.add_node(node, historical_mark=mark, module=new_module)
        self.graph.add_edge(nodes[n1], node)
        self.graph.add_edge(node, nodes[n2])

    def add_node_outside_edge(self, modules, historical_marker, supernet, supernets_map):
        nodes = list(nx.algorithms.topological_sort(self.graph))
        if len(nodes) == 1:
            return
        while (True):
            print("stuck in while loop,add_node_outside_edge ")
            n1 = randrange(0, len(nodes) - 1)
            n2 = randrange(n1, len(nodes))
            if ((nodes[n1], nodes[n2]) in self.graph.edges) and nodes[n1] != nodes[n2]:
                break

        new_module = choice(modules)
        self.module_id_generator()
        node = new_module.id + "_" + str(self.module_count)
        mark = historical_marker.mark_blueprint_genes()
        predecessor_mark = self.graph.nodes[nodes[n1]]["historical_mark"]
        supernets_map.update_map(mark, supernet,predecessor_mark)
        self.graph.add_node(node, historical_mark=mark, module=new_module)
        self.graph.add_edge(nodes[n1], node)
        self.graph.add_edge(node, nodes[n2])

    def add_node_at_out(self, modules, historical_marker, supernet, supernets_map):
        nodes = list(nx.algorithms.topological_sort(self.graph))
        new_module = choice(modules)
        self.module_id_generator()
        node = new_module.id + "_" + str(self.module_count)
        mark = historical_marker.mark_blueprint_genes()
        predecessor_mark = self.graph.nodes[nodes[-1]]["historical_mark"]
        supernets_map.update_map(mark, supernet,predecessor_mark)
        self.graph.add_node(node, historical_mark=mark, module=new_module)
        self.graph.add_edge(nodes[-1], node)


    def mutate(self, modules, historical_marker, supernet, supernets_map):

        selector = randint(1,5)
        if selector == 1:
            self.add_connection()
        if selector == 2:
            self.switch_module(modules)
        if selector == 3:
            self.add_node_in_edge(modules, historical_marker, supernet, supernets_map)
        if selector == 4:
            self.add_node_outside_edge(modules, historical_marker, supernet, supernets_map)
        if selector == 5:
            self.add_node_at_out(modules, historical_marker, supernet, supernets_map)

    def crossover(self, parent2, historical_marker):
        p1_graph = self.graph
        p2_graph = parent2.graph

        p1_nodes = list(nx.algorithms.topological_sort(p1_graph))
        p2_nodes = list(nx.algorithms.topological_sort(p2_graph))

        p1_nodes_marks = nx.get_node_attributes(p1_graph,"historical_mark")
        p2_nodes_marks = nx.get_node_attributes(p2_graph, "historical_mark")

        matching_nodes_p1 = []
        matching_nodes_p2 = []
        for k1,v1 in p1_nodes_marks.items():
            for k2,v2 in p2_nodes_marks.items():
                if v1 == v2:
                    matching_nodes_p1.append(k1)
                    matching_nodes_p2.append(k2)


        disjoint_nodes_p1 = list(set(p1_nodes) - set(matching_nodes_p1))
        disjoint_nodes_p2 = list(set(p2_nodes) - set(matching_nodes_p2))

        overlaps = []
        for n1,n2 in zip(matching_nodes_p1,matching_nodes_p2):
            selector = choice([True, False])
            if selector:
                overlaps.append((n1,p1_graph.nodes[n1]))
            else:
                overlaps.append((n2, p2_graph.nodes[n2]))

        disjoint1 = []
        disjoint2 = []

        for n in disjoint_nodes_p1:
            disjoint1.append((n,p1_graph.nodes[n]))
        for n in disjoint_nodes_p2:
            disjoint2.append((n, p2_graph.nodes[n]))

        def remap_edges(c_n, b):
            b_nodes_by_mark = {attr["historical_mark"]: node_id  for node_id , attr in c_n}
            remapped_edges = [(b_nodes_by_mark[b.nodes[u]["historical_mark"]]
                               , b_nodes_by_mark[b.nodes[v]["historical_mark"]]) for u, v in b.edges()]
            return remapped_edges

        if self.fitness() > parent2.fitness():
            nodes = overlaps + disjoint1
            edges = remap_edges(nodes,self.graph)

        else:
            nodes = overlaps + disjoint2
            edges = remap_edges(nodes, parent2.graph)

        child = build_dag(nodes, edges)

        child_blueprint = Blueprint(child,historical_marker.mark_blueprint())
        child_blueprint.id_generator()

        return child_blueprint

    def module_id_generator(self):
        self.module_count = self.module_count + 1


    def fitness(self):
        return self.fitness_values[1]

    def update_fitness(self, fitness_values):
        self.fitness_values = fitness_values

    def genotype_phenotype_features(self, historical_marker):
        network_size = 0
        network_depth = 0
        network_width = 0
        node_count = len(list(self.graph.nodes))
        edge_count = len(list(self.graph.edges))
        genetic_code = np.zeros(historical_marker.blueprint_marks)
        nodes = list(nx.algorithms.topological_sort(self.graph))
        simple_paths = list(nx.all_simple_paths(self.graph,nodes[0],nodes[-1]))
        path_lengths =  [len(path) for path in simple_paths]
        max_path = simple_paths[np.array(path_lengths).argmax()] if simple_paths else ["Nopath"]

        for node in nodes:
            module = self.graph.nodes[node]["module"]
            module_size, module_depth, module_width = module.get_network_size_and_depth()
            network_size = network_size + module_size
            network_width = network_width + module_width
            genetic_code[self.graph.nodes[node]["historical_mark"]] = 1
            if node in max_path:
                network_depth = network_depth + module_depth
        return  [node_count, edge_count, network_size, network_depth, network_width] #+ genetic_code.tolist()




