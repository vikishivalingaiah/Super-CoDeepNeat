import matplotlib.pyplot as plt
import networkx as nx
import numpy as np



def plot_supernet(supernet, show=False, pdfpages=None):
    nx.draw_networkx(supernet.graph)
    plt.title("Best supernet graph for " + supernet.id)
    if show:
        plt.show()
    if pdfpages != None:
        pdfpages.savefig()

    plt.close()


def plot_generation_best_fitnes(fitness_list, show=False, pdfpages=None):
    plt.subplot(1, 1, 1)
    plt.plot(list(range(-1,len(fitness_list) - 1)), fitness_list)
    plt.title("Best fitness vs generations")
    plt.xlabel("generation")
    plt.ylabel("Fitness values(Accuracy)")
    if show:
        plt.show()
    if pdfpages != None:
        pdfpages.savefig()
    plt.close()

def plot_generation_blueprint_fitnes(fitness_list, show=False, pdfpages=None):
    plt.subplot(1, 1, 1)
    plt.plot(list(range(0,len(fitness_list))), fitness_list)
    plt.title("fitness vs blueprints")
    plt.xlabel("nth blueprint in population")
    plt.ylabel("Fitness values(Accuracy)")
    if show:
        plt.show()
    if pdfpages != None:
        pdfpages.savefig()
    plt.close()

def plot_species_count(species_dict,show=False,pdfpages=None):
    for generation, species_details in species_dict.items():
        x = []
        y = []
        for speciesid, count in species_details.items():
            x.append(speciesid)
            y.append(count)

        plt.scatter(x,y)
        plt.title("Species count for generation" + str(generation))
        plt.xlabel("count")
        plt.ylabel("species id")
        if show:
            plt.show()
        if pdfpages != None:
            pdfpages.savefig()
        plt.close()

def plot_average_species_features(average_feature_dict, type, show=False, pdfpages=None):
    if type == "modules":
        features = ["size", "max_depth", "width"]
        splot = [2,2]
    elif type == "blueprints":
        features = ["node_count", "edge_count", "network_size", "network_depth", "network_width"]
        splot = [2,3]
    else:
        raise ValueError("type must be blueprints or modules")





    for n in range(1, len(features) +1):
        for key, value in average_feature_dict.items():
            if len(value["features"]) != 0:
                print(np.array(value["features"]))
                plt.plot(value["generations"],np.array(value["features"])[:,n - 1], ls="--", label=key)
                plt.ylabel(features[n-1])
                plt.xlabel("generations")
            plt.legend()
            plt.title("Species features over generation for" + type)

        if show:
            plt.show()
        if pdfpages != None:
            pdfpages.savefig()
        plt.close()


def plot_blueprints(blueprint,title,show=False,pdfpages=None,root_path="./"):
    
    if show:
        nx.draw_networkx(blueprint.graph)
        plt.title(title + "blueprint graph for " + blueprint.id)    
        plt.savefig(root_path + "/src/test_images/" + title + str(blueprint.id))
    if pdfpages != None:
        nx.draw_networkx(blueprint.graph)
        plt.title(title + "blueprint graph for " + blueprint.id)    
        pdfpages.savefig()

    plt.close()


    graph_dict = {}
    for node in nx.algorithms.topological_sort(blueprint.graph):
        graph_dict[node] = {}
        module = blueprint.graph.nodes[node]["module"]
        module_nodes = []
        module_edges = []

        historical_mark = blueprint.graph.nodes[node]["historical_mark"]
        for module_node in list(module.graph.nodes):
            actual_node = module_node + "_"  +str(historical_mark)
            module_nodes.append(actual_node)

        for u,v in list(module.graph.edges):
            u = u + "_" + str(historical_mark)
            v = v + "_" + str(historical_mark)
            module_edges.append((u,v))

        topological_nodes = list(nx.algorithms.topological_sort(module.graph))
        graph_dict[node]["origin"] = topological_nodes[0] +  "_"  + str(historical_mark)
        graph_dict[node]["termin"] = topological_nodes[-1] +  "_"  + str(historical_mark)
        graph_dict[node]["module_nodes"] = module_nodes
        graph_dict[node]["module_edges"] = module_edges

    blueprint_specific_edges = []
    for u,v in list(blueprint.graph.edges):
        u = graph_dict[u]["termin"]
        v = graph_dict[v]["origin"]
        blueprint_specific_edges.append((u,v))

    final_nodes = []
    final_edges = []

    for node, child_graph in graph_dict.items():
        final_nodes = final_nodes + child_graph["module_nodes"]
        final_edges = final_edges + child_graph["module_edges"]

    final_edges = final_edges + blueprint_specific_edges

    g = nx.DiGraph()
    g.add_nodes_from(final_nodes)
    g.add_edges_from(final_edges)

    if show:
        nx.draw_networkx(g)
        plt.title(title + "expanded blueprint graph for " + blueprint.id)
        plt.savefig(root_path + "/src/test_images/" + title + "EXPANDED"+ str(blueprint.id))
    if pdfpages != None:
        nx.draw_networkx(g)
        plt.title(title + "expanded blueprint graph for " + blueprint.id)
        pdfpages.savefig()
    plt.close()






