import networkx as nx
from src.training.models import ModuleModel,ComponentModel,SupernetModel
import tensorflow as tf
import matplotlib.pyplot as plt
from src.training.models import BlueprintModel
import time
import logging
from datetime import datetime
import json

class Trainer():
    def __init__(self, train_dataset, val_dataset, test_dataset, input_layer, output_layer, optimizer, loss_fn, metrics, epochs=1,root_path="./"):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.epochs = epochs
        self.root_path = root_path

    def train_supernets_by_blueprints(self, blueprints, supernets_map, generation, save=False):
        train_map = {}
        fitness_map = {}
        path_map = {}
        i = 0
        logging.info("*" * 40)
        for blueprint in blueprints:
            logging.info("Training " + str(i) + "th blueprint with id: " + str(blueprint.id))
            #nx.draw_networkx(blueprint.graph)
            #plt.show()
            blueprint_model = BlueprintModel(blueprint, supernets_map)
            x = self.input_layer
            x = blueprint_model.call(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(256,activation="relu")(x)
            x = tf.keras.layers.Dense(10,activation="softmax")(x)#self.output_layer(x)
            train_model = tf.keras.Model(self.input_layer,x)
            #train_model.summary()
            train_model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=self.metrics)
            history = train_model.fit(self.train_dataset,epochs=self.epochs,validation_data=self.val_dataset)
            results = self.fit_blueprints(train_model)
            logging.info(results)
            fitness_map[blueprint.id] = results
            train_map[blueprint.id] = history
            if save:
                timestamp = datetime.now().strftime('_%H_%M_%d_%m_%Y')
                path = self.root_path + "/saved_models/" + "model" + str(
                    blueprint.id) + str(generation) + timestamp
                tf.keras.utils.plot_model(train_model,
                                          to_file= self.root_path + "/src/test_images/blueprint_model_" + str(
                                              blueprint.id) + str(generation) + timestamp + ".png", show_layer_names=True, show_shapes=True)
                path_map[blueprint.id] = path
                train_model.save(path)
                logging.info("Saved model, image and json for  blueprint" + blueprint.id)
                logging.info("At path " + path)
            i = i+1
            logging.info("Training for model complete")
        logging.info("*"*40)
        if save:
            return train_map, fitness_map, path_map
        else:
            return train_map, fitness_map

    def save_blueprint_model(self, blueprints, supernets_map, supernet, generation):
        train_map = {}
        fitness_map = {}
        i = 0
        logging.info("*" * 40)
        for blueprint in blueprints:
            logging.info("Building model for " + str(i) + "th blueprint with id: " + str(blueprint.id))
            # nx.draw_networkx(blueprint.graph)
            # plt.show()
            blueprint_model = BlueprintModel(blueprint, supernets_map)
            x = self.input_layer
            x = blueprint_model.call(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(256,activation="relu")(x)
            x = tf.keras.layers.Dense(10,activation="softmax")(x)
            #x = self.output_layer(x)
            train_model = tf.keras.Model(self.input_layer, x)
            # train_model.summary()
            timestamp = datetime.now().strftime('_%H_%M_%d_%m_%Y')
            train_model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=self.metrics)
            tf.keras.utils.plot_model(train_model,
                                      to_file= self.root_path + "/src/test_images/blueprint_model_" + str(blueprint.id) + str(
                                          generation) + timestamp + ".png", show_layer_names=True, show_shapes=True)
            blueprint_info = {}
            for node in nx.topological_sort(blueprint.graph):
                blueprint_info[node] = {}
                module = blueprint.graph.nodes[node]["module"]

                for module_node in nx.topological_sort(module.graph):
                    activations = supernet.graph.nodes[module_node]["activations"]
                    components = supernet.graph.nodes[module_node]["component_list"]
                    decoded_components = {}
                    for component in components:
                        basic_layers = []
                        for component_node in nx.topological_sort(component.layer_graph):
                            nn_layer = component.layer_graph.nodes[component_node]["layer"]
                            basic_layers.append((nn_layer.keras_layer, nn_layer.keras_hyperparameters))
                        decoded_components[component.id] = basic_layers


                    blueprint_info[node][module_node] = {}
                    blueprint_info[node][module_node]["activations"] = activations
                    blueprint_info[node][module_node]["components"] = decoded_components


            with open(self.root_path + "/src/test_images/blueprint_model_"
                      + str(blueprint.id) + str(generation) + timestamp + ".json", 'w') as outfile:
                json.dump(blueprint_info, outfile, indent=4)
            path = self.root_path + "/saved_models/"+ "model" + str(blueprint.id) + str(generation)+ timestamp
            train_model.save(path)
            logging.info("Saved model, image and json for best blueprint" + blueprint.id)
            logging.info("At path " + path)
            i = i + 1

        logging.info("*" * 40)
        return path

    def fit_blueprints(self, blueprint_model):
        results = blueprint_model.evaluate(self.val_dataset,verbose=1)
        print(results)
        return results

    def validate_test(self, path):
        logging.info("=" * 40)
        model = tf.keras.models.load_model(path)
        logging.info("Validating model at ")
        logging.info(path)
        dot_img_file = self.root_path + "/src/test_images/" + path.split("/")[
            -1] + ".png"
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
        #model.summary()

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
        optimizer = "SGD"

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
        results = model.evaluate(self.test_dataset, verbose=1)
        logging.info(results)
        logging.info("=" * 40)
        return results



 