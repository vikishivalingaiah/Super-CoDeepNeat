import networkx as nx
from random import sample

class Component:
    def __init__(self,layer_graph,name=None,identifier=None):
        self.name = name
        self.id = identifier
        if (nx.is_directed_acyclic_graph(layer_graph)):
            self.layer_graph = layer_graph
        else:
            print("The given graph for component is DAG: %s" % nx.is_directed_acyclic_graph(layer_graph))


    def id_generator(self):
        unique_number = "".join(sample([str(i) for i in range(0, 9)], 6))
        self.id = "component" + "-" + unique_number
        return unique_number

    """
    def mutate_hyperparameter(self):
    ## Mutate the existing hyperparameters and set the new hyperparameters to return this component
        for hyperparameter in self.component.tf.keras.hyperparameters:
            if hyperparameter is categorical:
                current_value = hyperparameter
                new_value = random_categorical_operator(hyperparameter_table[hyperparameter])

            else if hyperparameter is continuous:
                current_value = hyperparameter
                new_value = random_continuous_operator(hyperparameter_table[hyperparameter])
    """



