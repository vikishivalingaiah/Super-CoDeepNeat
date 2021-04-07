from random import  sample,randint
import networkx as nx



class Supernet:
    def __init__(self, graph,  name=None, identifier=None):
        self.name = name
        self.id = identifier
        if (nx.is_directed_acyclic_graph(graph)):
            self.graph = graph
        else:
            print("The given graph is DAG: %s" % nx.is_directed_acyclic_graph(graph))

    def id_generator(self):
        unique_number = "".join(sample([str(i) for i in range(0, 9)], 6))
        self.id = "supernet" + "-" + unique_number
        return unique_number

    def update_channels_count(self, C_in,C_out):
        count = len(list(nx.algorithms.topological_sort(self.graph)))
        #print(C_in)
        #print(C_out)
        incr = (C_out - C_in) // count
        C_out_node = 0
        for node in nx.algorithms.topological_sort(self.graph):
            if C_out_node < C_out_node:
                C_out_node = C_out_node + incr
            else:
                C_out_node = C_out
                if C_out_node > 256:
                    C_out_node = randint(1,256)
            for component in self.graph.nodes[node]["component_list"]:
                for component_node in nx.algorithms.topological_sort(component.layer_graph):
                    component.layer_graph.nodes[component_node]["layer"].keras_hyperparameters["filters"] = C_out_node
