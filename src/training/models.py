import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
import time
from src.training.custom_layers import ConvBNRelu,ConvAdjustChannels

class ZeroLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()


    def call(self, inputs, training=None):
        x = tf.math.scalar_mul(0.0, inputs, name=None)
        return x



class ComponentModel(tf.keras.Model):
    def __init__(self,component):
        super().__init__()
        self.component = component
        self.basic_layers = []
        for node in nx.topological_sort(component.layer_graph):
            nn_layer = component.layer_graph.nodes[node]["layer"]
            self.basic_layers.append(eval(nn_layer.keras_layer)(**nn_layer.keras_hyperparameters))

    def call(self, inputs, training=None):
        x = inputs
        print(x.shape)
        for basic_layer in self.basic_layers:
            x = basic_layer.call(x)
        return x

class ModuleModel(tf.keras.Model):
    def __init__(self,module, supernet):
        super().__init__()
        self.module = module
        self.component_models = []
        for node in nx.topological_sort(module.graph):
            component = module.graph.nodes[node]["component"]
            self.component_models.append(ComponentModel(component))

    def call(self, inputs, training=None):
        x = inputs
        for component_model in self.component_models:
            x = component_model.call(x)
        return x



class SupernetModel(tf.keras.Model):
    def __init__(self, supernet):
        super().__init__()
        self.supernet = supernet
        self.component_models = {}
        for node in nx.topological_sort(self.supernet.graph):
            self.component_models[node] = []
            components = supernet.graph.nodes[node]["component_list"]
            for component in components:
                self.component_models[node].append(ComponentModel(component))

    def call(self, inputs, module=None, training=None):
        concat_axis = 3
        nodes_outputs = {}
        x = inputs
        if module is None:
            graph = self.supernet.graph
        else:
            graph = module.graph
        print(list(nx.topological_sort(graph)))
        for node in nx.topological_sort(graph):
            predecessors = list(graph.predecessors(node))
            if predecessors:
                local_x = []
                if len(predecessors) > 1:
                    for predecessor in predecessors:
                        local_x.append(nodes_outputs[predecessor])
                    x =  tf.keras.layers.Add()(local_x)
                else:
                    x = nodes_outputs[predecessors[0]]
            else:
                if type(x) is list:
                    x = (inputs,concat_axis)
                else:
                    x = inputs

            #node_outputs = []
            i = 0
            print(node)
            print(graph.nodes[node]["activations"])
            for component in self.component_models[node]:
                #op = component.call(x)
                activated = graph.nodes[node]["activations"][i]
                if activated:
                    print(node)
                    x = component.call(x)
                    #node_outputs.append(op)
                #else:
                #    node_outputs.append(ZeroLayer()(op))
                i = i + 1

            #x = tf.concat(node_outputs,concat_axis)
            nodes_outputs[node] = x
        return x

class BlueprintModel(tf.keras.Model):
    def __init__(self, blueprint ,supernets_map):
        super().__init__()
        self.blueprint = blueprint
        self.supernets_map = supernets_map
        self.module_map = {}
        for node in nx.algorithms.topological_sort(self.blueprint.graph):
            self.module_map[node] = {}
            supernet_model = supernets_map.map[self.blueprint.graph.nodes[node]["historical_mark"]]["supernet_model"]
            self.module_map[node]["supernet_model"] = supernet_model
            self.module_map[node]["module"] = self.blueprint.graph.nodes[node]["module"]



    def call(self, inputs):
        ip_op_map = {}
        for node in  nx.algorithms.topological_sort(self.blueprint.graph):
            ip_op_map[node] = {}
            historical_mark = self.blueprint.graph.nodes[node]["historical_mark"]
            C_in = self.supernets_map.map[historical_mark]["C_in"]
            if self.blueprint.graph.in_degree(node) == 0:
                x = inputs
            else:
                all_inputs = []
                predecessors = list(self.blueprint.graph.predecessors(node))
                for predecessor in predecessors:
                    historical_mark = self.blueprint.graph.nodes[predecessor]["historical_mark"]
                    C_out_predecessor = self.supernets_map.map[historical_mark]["C_out"]
                    if C_in != C_out_predecessor:
                       predecessor_op = ConvAdjustChannels(C_in)(ip_op_map[predecessor]["output"])
                    else:
                        predecessor_op = ip_op_map[predecessor]["output"]
                    all_inputs.append(predecessor_op)
                if len(predecessors) > 1:
                    x =  tf.keras.layers.Add()(all_inputs)
                else:
                    x = all_inputs[0]
            x = self.module_map[node]["supernet_model"].call(x,self.module_map[node]["module"])
            ip_op_map[node]["output"] = x
        t2 = time.perf_counter()
        return x

















