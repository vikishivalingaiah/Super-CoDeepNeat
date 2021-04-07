from random import  choice,randint,uniform,sample
class Layer:
    def __init__(self, layer, hyperparameter_table,name=None, identifier=None,hyperparameters=None,):
        self.name = name
        self.id = identifier
        self.keras_layer = layer
        self.keras_hyperparameters = hyperparameters
        self.keras_hyperparameters_table = hyperparameter_table

    def set_random_hyperparameters(self):
        hyperprameter_config = {}
        for hyperprameter_name, config_values in self.keras_hyperparameters_table.items():
            if config_values["type"] == "list":
                hyperprameter_config[hyperprameter_name] = choice(config_values["values"])
            else:
                if config_values["type"] == "int":
                    hyperprameter_config[hyperprameter_name] = randint(config_values["values"][0],
                                                                       config_values["values"][1])
                if config_values["type"] == "float":
                    hyperprameter_config[hyperprameter_name] = uniform(config_values["values"][0],
                                                                       config_values["values"][1])

        self.keras_hyperparameters = hyperprameter_config

    def id_generator(self):
        unique_number = "".join(sample([str(i) for i in range(0,9)],6))
        self.id = "layer" + "-" + unique_number
        return unique_number

