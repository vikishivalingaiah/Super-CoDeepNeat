from random import sample, randint, choice
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
from src.encoding.DAGEncoder import build_dag
import numpy as np
import logging


class Module:
    def __init__(self, graph, supernet, mark, name=None, identifier=None):
        self.name = name
        self.id = identifier
        self.mark = mark
        self.supernet = supernet
        self.cooperative_fitness_values = []
        self.fitness_values = [99,0]
        self.species = -1

        if (nx.is_directed_acyclic_graph(graph)):
            self.graph = graph
        else:
            print("The given graph is DAG: %s" % nx.is_directed_acyclic_graph(graph))


    def id_generator(self):
        unique_number = "".join(sample([str(i) for i in range(0, 9)], 6))
        self.id = "module" + "-" + unique_number
        return unique_number

    def random_bit_flip(self,bitstring):

        if bitstring == None:
            raise ValueError("Illegal value for bitstring")
        if len(bitstring) < 2:
            logging.debug("single component case, no bitflip required")
            return


        random_bit = 0
        loc = 0
        for i in range(len(bitstring)):
            if bitstring[i] == 1:
                bitstring[i] = 0
                random_bit = i
                loc = i
                break

        while(random_bit == loc):
            print("Stuck in random_bitflip, modules")
            random_bit = randint(0,len(bitstring) - 1)
        bitstring[random_bit] = 1
        return bitstring

    def mutate_switch_activation(self):
        node_list = [node for node in list(self.graph.nodes())]
        if node_list:
            node = choice(node_list)
            self.graph.nodes[node]["activations"] = self.random_bit_flip(
                self.graph.nodes[node]["activations"])

    def mutate_by_adding_edge_from_supernet(self):

        module_graph = self.graph
        supernet_graph = self.supernet.graph

        node_list = [node for node in list(nx.algorithms.topological_sort(module_graph)) if
                     module_graph.out_degree(node) != 0]
        if len(node_list) >= 2:
            u = choice(node_list)
            out_edges = [(u_1, v_1) for (u_1, v_1) in supernet_graph.out_edges(u) if
                         (v_1 in list(module_graph.nodes) and (u_1, v_1) not in list(module_graph.edges))]
            if out_edges:
                edge = choice(out_edges)
                edges = list(module_graph.edges)
                edges.append(edge)
                new_module_graph = build_dag(module_graph.nodes(data=True), edges)
            else:
                new_module_graph = module_graph
        else:
            new_module_graph = module_graph

        node_list = [node for node in list(new_module_graph.nodes())]#if new_module_graph.out_degree(node) != 0 and new_module_graph.in_degree(node) != 0]
        if node_list:
            node = choice(node_list)
            new_module_graph.nodes[node]["activations"] = self.random_bit_flip(new_module_graph.nodes[node]["activations"])

        self.graph = new_module_graph

    def mutate_by_adding_nodes_from_supernet(self):

        module_graph = self.graph
        supernet_graph = self.supernet.graph

        difference_nodes = [node for node in list(supernet_graph.nodes(data=True)) if
                            node not in list(module_graph.nodes(data=True))]
        if difference_nodes:
            add_node = choice(difference_nodes)
        else:
            return

        current_node = add_node
        out_edges = []

        nodes_list = list(module_graph.nodes(data=True))
        nodes_list.append(current_node)
        while supernet_graph.out_degree(current_node[0]) != 0:
            print("Stuck in while loop mutate_by_adding_nodes_from_supernet")
            out_edge = choice(list(supernet_graph.out_edges(current_node[0])))
            if out_edge[1] in list(module_graph.nodes):
                attr = module_graph.nodes[out_edge[1]]
            else:
                attr = supernet_graph.nodes[out_edge[1]]

            current_node = (out_edge[1], attr)
            nodes_list.append(current_node)
            out_edges.append(out_edge)

        current_node = add_node
        in_edges = []

        while supernet_graph.in_degree(current_node[0]) != 0:
            print("Stuck in while loop mutate_by_adding_nodes_from_supernet")
            in_edge = choice(list(supernet_graph.in_edges(current_node[0])))
            if in_edge[0] in list(module_graph.nodes):
                attr = module_graph.nodes[in_edge[0]]
            else:
                attr = supernet_graph.nodes[in_edge[0]]
            current_node = (in_edge[0], attr)
            nodes_list.append(current_node)
            in_edges.append(in_edge)

        edges = list(module_graph.edges)
        edges = list(set(edges + in_edges + out_edges))
        new_module_graph = build_dag(nodes_list, edges)

        node_list = [node for node in list(new_module_graph.nodes())]
                     #if new_module_graph.out_degree(node) != 0 and new_module_graph.in_degree(node) != 0]
        if node_list:
            node = choice(node_list)
            new_module_graph.nodes[node]["activations"] = self.random_bit_flip(new_module_graph.nodes[node]["activations"])

        self.graph = new_module_graph


    def mutate_by_removing_edges_from_supernet(self):
        module_graph = self.graph
        supernet_graph = self.supernet.graph

        node_list = [node for node in list(nx.algorithms.topological_sort(module_graph)) if
                     module_graph.out_degree(node) != 0]

        module_nodes = list(nx.algorithms.topological_sort(module_graph))

        if len(node_list) >= 2:
            iter = 0
            found_edge = False
            edges = list(module_graph.edges)
            while True:
                print("Stuck in while loop mutate_by_removing_edges_from_supernet")
                if iter == 50:
                    break
                random_edge = choice(edges)
                u = random_edge[0]
                v = random_edge[1]

                supernet_successors = [sucessor_supernet for sucessor_supernet in list(supernet_graph.successors(u))
                                     if sucessor_supernet in module_nodes]
                supernet_predecessors = [predecessor_supernet for predecessor_supernet in
                                     list(supernet_graph.predecessors(v)) if predecessor_supernet in module_nodes]

                if (len(list(module_graph.successors(u))) > 1 or len(supernet_successors) > 1)\
                        and (len(list(module_graph.predecessors(v))) > 1 or len(supernet_predecessors) > 1):
                    found_edge = True
                    break

                iter = iter + 1

            if found_edge:
                u_successors = list(module_graph.successors(u)) if len(list(module_graph.successors(u))) > 1 \
                               else [sucessor_supernet for sucessor_supernet in list(supernet_graph.successors(u))
                                     if sucessor_supernet in module_nodes]
                v_predecessors = list(module_graph.predecessors(v)) if len(list(module_graph.predecessors(v))) > 1 \
                               else [predecessor_supernet for predecessor_supernet in
                                     list(supernet_graph.predecessors(v)) if predecessor_supernet in module_nodes]

                u_successors.remove(v)
                v_predecessors.remove(u)


                if len(list(module_graph.successors(u)))  <= 1 and len(u_successors) > 0:
                   v_1 = choice(u_successors)
                   edges.append((u,v_1))

                if len(list(module_graph.predecessors(v))) <= 1 and len(v_predecessors) > 0:
                    u_1 = choice(v_predecessors)
                    edges.append((u_1, v))

                edges.remove((u, v))

                new_module_graph = build_dag(module_graph.nodes(data=True), edges)

            else:
                new_module_graph = module_graph

            node_list = [node for node in list(new_module_graph.nodes())]
            if node_list:
                node = choice(node_list)
                new_module_graph.nodes[node]["activations"] = self.random_bit_flip(
                    new_module_graph.nodes[node]["activations"])

            self.graph = new_module_graph


    def mutate_by_removing_nodes(self):
        module_graph = self.graph
        supernet_graph = self.supernet.graph
        found_node = False

        nodes = [node for node in list(nx.algorithms.topological_sort(module_graph)) if
                                  module_graph.out_degree(node) != 0 and module_graph.in_degree(node) != 0]
        module_nodes = list(nx.algorithms.topological_sort(module_graph))
        if nodes:
            iter = 0
            while True:
                if iter == 50:
                    break

                remove_node = choice(nodes)
                predecessors = list(module_graph.predecessors(remove_node))
                successors = list(module_graph.successors(remove_node))

                predecessor_full_path_count = 0
                for predecessor in predecessors:
                    successors_p = [successor_node for successor_node in supernet_graph.successors(predecessor)
                                    if successor_node in module_nodes]
                    successors_p.remove(remove_node)

                    if len(successors_p) > 0:
                        predecessor_full_path_count = predecessor_full_path_count + 1

                successor_full_path_count = 0
                for successor in successors:
                    predecessors_s = [predecessor_node for predecessor_node in supernet_graph.predecessors(successor)
                                    if predecessor_node in module_nodes]
                    predecessors_s.remove(remove_node)

                    if len(predecessors_s) > 0:
                        successor_full_path_count = successor_full_path_count + 1


                if successor_full_path_count == len(successors) and predecessor_full_path_count == len(predecessors):
                    found_node = True
                    break

                iter = iter + 1

            if found_node:
                module_nodes.remove(remove_node)
                predecessors = list(module_graph.predecessors(remove_node))
                successors = list(module_graph.successors(remove_node))

                edges = list(module_graph.edges)

                for predecessor in predecessors:
                    edges.remove((predecessor,remove_node))
                    successors_p = [successor_node for successor_node in supernet_graph.successors(predecessor)
                                    if successor_node in module_nodes]
                    #successors_p.remove(remove_node)

                    successor_p = choice(successors_p)
                    edges.append((predecessor,successor_p))

                for successor in successors:
                    edges.remove((remove_node, successor))
                    predecessors_s = [predecessor_node for predecessor_node in supernet_graph.predecessors(successor)
                                      if predecessor_node in module_nodes]
                   # predecessors_s.remove(remove_node)
                    predecessor_s = choice(predecessors_s)
                    edges.append((predecessor_s, successor))

                final_nodes = [node for node in module_graph.nodes(data=True) if node[0] in module_nodes]

                new_module_graph = build_dag(final_nodes, edges)
            else:
                new_module_graph = module_graph

            node_list = [node for node in list(new_module_graph.nodes())]
            if node_list:
                node = choice(node_list)
                new_module_graph.nodes[node]["activations"] = self.random_bit_flip(
                    new_module_graph.nodes[node]["activations"])

            self.graph = new_module_graph

    def crossover(self, parent2, historical_marker):


        p1 = self.graph
        p2 = parent2.graph

        p1_nodes = set(p1.nodes())
        p2_nodes = set(p2.nodes())

        matching_nodes = p1_nodes.intersection(p2)
        disjoint1_nodes = p1_nodes.difference(p2_nodes)
        disjoint2_nodes = p2_nodes.difference(p1_nodes)

        nodes_list = []

        for node in matching_nodes:
            selector = choice([True,False])
            if selector:
                nodes_list.append((node,p1.nodes[node]))
            else:
                nodes_list.append((node, p2.nodes[node]))


        p1_edges = set(p1.edges)
        p2_edges = set(p2.edges)

        matching = p1_edges.intersection(p2_edges)
        disjoint1 = p1_edges.difference(p2_edges)
        disjoint2 = p2_edges.difference(p1_edges)

        if self.fitness() > parent2.fitness():
            edges = matching.union(disjoint1)
            nodes_list = nodes_list + [(node, p1.nodes[node]) for node in disjoint1_nodes]
        else:
            edges = matching.union(disjoint2)
            nodes_list = nodes_list + [(node, p2.nodes[node]) for node in disjoint2_nodes]

        child = build_dag(nodes_list, edges)

        child_module = Module(child, self.supernet, historical_marker.mark_module())
        child_module.id_generator()

        return child_module


    def mutate(self):
        selector = randint(1, 5)
        if selector == 1:
            self.mutate_by_adding_nodes_from_supernet()
        if selector == 2:
            self.mutate_by_adding_edge_from_supernet()
        if selector == 3:
            self.mutate_by_removing_nodes()
        if selector == 4:
            self.mutate_by_removing_edges_from_supernet()
        if selector == 5:
            self.mutate_switch_activation()


    def fitness(self):
        return self.fitness_values[1]

    def update_cooperative_fitness_values(self,fitness_values):
        self.cooperative_fitness_values.append(fitness_values)

    def update_shared_fitness(self, species_size=1):
        if len(self.cooperative_fitness_values) >= 1:
            loss = np.array(self.cooperative_fitness_values)[:, 0].mean() / species_size
            acc = np.array(self.cooperative_fitness_values)[:, 1].mean() / species_size
            self.fitness_values = [loss, acc]

    def get_network_size_and_depth(self):
        size = self.graph.size()
        nodes = list(nx.algorithms.topological_sort(self.graph))
        simple_paths = list(nx.all_simple_paths(self.graph, nodes[0], nodes[-1]))
        path_lengths = [len(path) for path in simple_paths]
        max_depth = np.array(path_lengths).max()
        width = len(simple_paths)

        return size, max_depth, width

    def genotype_phenotype_features(self, historical_marker):
        size, max_depth, width = self.get_network_size_and_depth()

        return [size, max_depth, width]
