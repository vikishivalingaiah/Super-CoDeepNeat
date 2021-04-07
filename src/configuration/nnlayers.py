def initialize_data(filename=None):
    if filename is None:
        layers = [ {
            "tf.keras.layer": "tf.keras.layers.Dense",
            "tf.keras.hyperparameters_table": {
                "units": {
                    "values": [1, 256],
                    "type": "int"
                },
                "activation": {
                    "values": ["relu", "tanh", "sigmoid"],
                    "type": "list"
                }
            }
        }
        ]
        additional_layers = []

        sequential_components = []

        """
        supernet_config = [{
            "tf.keras.layer": "tf.keras.layers.Dense",
            "tf.keras.hyperparameters_table": {
                "units": {
                    "values": [50, 20],
                    "type": "list"
                },
                "activation": {
                    "values": ["relu", "tanh", "sigmoid"],
                    "type": "list"
                    }
                }
            }
        ]

        """
        supernet_config = [{
           "tf.keras.layer": "ConvBNRelu",#"tf.keras.layers.Conv2D",
           "tf.keras.hyperparameters_table": {
               "filters": {
                   "values": [4],
                   "type": "list"
               },
               "kernel_size":{
                   "values": [3,5],
                   "type": "list"
               },
               "pool_size":{
                   "values": [2,3],
                   "type": "list"
               },
               "maxpool":{
                   "values": [True, False],
                   "type": "list"
               },
               "padding":{
                   "values": ["same"],
                   "type": "list"
                   },
               "activation": {
                   "values": ["relu"],
                   "type": "list"
                   }
               }
           }]


    return layers, additional_layers, sequential_components, supernet_config

def get_evolution_parameters(filename=None):
    if filename is None:
        config = {
            "generations" : 20,
            "supernet" : {
                "population_size" : 1,
                "minimal_individual_sizes" : [5]
            },
            "modules" : {
                "num_of_species": 3,
                "population_size" : 20,
                "delta": 0.1,
                "minimal_individual_sizes": [1],
                "generation_gap": 0.5,
                "elitism_rate": 0.3,
                "crossover_probability": 0.9,
                "mutation_rate": 0.9
            },
            "blueprints": {
                "num_of_species": 3,
                "population_size": 20,
                "delta" : 0.7,
                "minimal_individual_sizes": [1],
                "generation_gap": 0.5,
                "elitism_rate": 0.3,
                "crossover_probability": 0.9,
                "mutation_rate": 0.9
            },
        }

    return config
