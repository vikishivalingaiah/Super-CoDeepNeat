import networkx as nx
import matplotlib.pyplot as plt
from random import randrange, sample



def build_dag(nodes, edges=None):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    if edges != None:
        G.add_edges_from(edges)

    if not (nx.is_directed_acyclic_graph(G)):
        raise ValueError("The formed graph is not a DAG")

    out_nodes = [node for node in nx.topological_sort(G) if G.out_degree(node) ==0]
    in_nodes = [node for node in nx.topological_sort(G) if G.in_degree(node) ==0]
    if len(out_nodes) == 1 and len(in_nodes) == 1:
        return G
    else:
        raise ValueError("Found multiple inputs or outputs")

class DAGEncoder:
    def __init__(self):
        self.node_size = 0
        self.G = nx.DiGraph()
        self.hidden_nodes = list(self.G.nodes)

    def generate_minimal_dag(self, node_size):

        self.node_size = node_size
        # self.G.add_node("Input")
        # self.G.add_node("output")
        random_nodes = sample(range(10), node_size)
        random_nodes.sort()
        print(random_nodes)
        self.G.add_nodes_from(random_nodes)
        self.hidden_nodes = list(self.G.nodes)

        # self.G.add_edge(list(G.nodes)[-1], "output")
        # self.G.add_edge("Input", list(G.nodes)[0])
        # hidden_nodes = set(self.G.nodes) - set(["Input", "output"])

        self.G.add_edges_from(
            [(self.hidden_nodes[i], self.hidden_nodes[i + 1]) for i in range(len(self.hidden_nodes) - 1)])

    def generate_from_edges(self,edges: list):
        self.G.add_edges_from(edges)
        self.node_size = len(list(self.G.nodes))
        self.hidden_nodes = list(self.G.nodes)

    def draw(self):
        plt.subplot(121)
        nx.draw(self.G, with_labels=True, font_weight='bold')
        plt.show()

    def mutate_connection(self):
        n1 = randrange(0, self.node_size)
        n2 = randrange(n1, self.node_size)
        print("n1,n2 for connection %s, %s" % (n1, n2))
        if (n2 > n1) and (n1 in self.hidden_nodes) and (n2 in self.hidden_nodes) and ((self.hidden_nodes[n1], self.hidden_nodes[n2]) not in self.G.edges):
            print("connection")
            self.G.add_edge(n1, n2)

    def mutate_node(self):
        n1 = randrange(0, self.node_size - 1)
        n2 = n1 + 1
        print("n1,n2 for connection %s, %s" % (n1, n2))
        if (self.hidden_nodes[n1], self.hidden_nodes[n2]) in self.G.edges:
            print("Mutation")
            self.G.remove_edge(self.hidden_nodes[n1], self.hidden_nodes[n2])
            while True:
                new_node = randrange(0, self.node_size)
                if new_node not in self.hidden_nodes:
                    break

            self.G.add_node(new_node)
            self.G.add_edge(self.hidden_nodes[n1], new_node)
            self.G.add_edge(new_node, self.hidden_nodes[n2])
            self.node_size = self.node_size + 1

    def crossover(self, dag: 'DAGEncoder') -> 'DAGEncoder':
       s1 = set(self.G.edges)
       s2 = set(dag.G.edges)

       overlap = s1.intersection(s2)
       s1_difference = s1 - overlap
       s2_differene = s2 - overlap
       print(s1_difference)
       print(s2_differene)
       offspring = overlap.union(s1_difference)
       new_offspring = DAGEncoder()
       new_offspring.generate_from_edges(list(offspring))
       return  new_offspring

