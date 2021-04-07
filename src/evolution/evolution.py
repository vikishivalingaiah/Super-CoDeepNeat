from src.evolution.initialization import Initialization
from src.evolution.historical_marker import HistoricalMarker
from src.evolution.historical_supernets import HistoricalSupernetMap
from src.encoding.blueprint import Blueprint
from src.encoding.module import Module
from random import choice, uniform
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
from src.evolution.species import Species
from sklearn.neighbors import NearestCentroid
from numpy import linalg
from sklearn.preprocessing import scale
import copy
import networkx as nx
import logging
import time
import json
import shutil
from src.utils import pdfsummary


class Evolution:
    def __init__(self, config_provider, trainer, blueprints=[], modules=[], supernets=[], pdfpages=None,root_path="./"):
        self.config_provider = config_provider
        self.trainer = trainer
        self.historical_marker = HistoricalMarker()
        self.supernets_map = HistoricalSupernetMap()
        self.modules = modules
        self.blueprints = blueprints
        self.supernets = supernets
        self.best_supernet = None
        self.fitness_map_blueprints = {}
        self.training_map_blueprints = {}
        self.fitness_map_modules = {}
        self.blueprint_species_list = []
        self.module_species_list = []
        self.intermediate_blueprint_population = []
        self.intermediate_module_population = []
        self.pdfpages = pdfpages
        self.root_path = root_path

    def evolve(self):
        t1 = time.perf_counter()
        logging.info("=" * 80)
        logging.info("Evolution Algorithm")
        logging.info("=" * 80)
        logging.info("Beginning evolution........")
        n_best_blueprint = []
        saved_paths = []
        saved_paths_original = []
        evo_parameters = self.config_provider.get_evolution_parameters()
        module_species_average_features = {}
        blueprint_species_average_features = {}
        logging.info("Using parameters:")
        logging.info(evo_parameters)
        logging.info("Initializing...............")
        self.initialization(evo_parameters)
        logging.info("Initialization complete.")
        logging.info("Calculating fitness.....")
        for blueprint in self.blueprints:
            pdfsummary.plot_blueprints(blueprint, "generation" + "initialization", show=True,root_path = self.root_path)#pdfpages=self.pdfpages)
        path_map = self.fit_blueprints(-1)
        fitness_blueprints = []
        for key, value in self.fitness_map_blueprints.items():
            fitness_blueprints.append(value[-1])
        pdfsummary.plot_generation_blueprint_fitnes(fitness_blueprints, pdfpages=self.pdfpages)
        self.fit_modules()
        self.fit_species()
        blueprin_species_dict = {}
        module_species_dict = {}
        pdfsummary.plot_supernet(self.best_supernet,self.pdfpages)
        best_blueprint = self.get_best_blueprint()
        self.clean_up(path_map,best_blueprint)
        n_best_blueprint.append(best_blueprint.fitness())
        self.trainer.save_blueprint_model([best_blueprint],self.supernets_map, self.best_supernet, -1)
        logging.info("Best blueprint id " + str(best_blueprint.id) + " with fitness " + str(
            best_blueprint.fitness()) + " for random intialization")
        t2 = time.perf_counter()
        for species in self.module_species_list:
            if species.id not in list(module_species_average_features.keys()):
                module_species_average_features[species.id] = {"generations":[],"features":[]}
            else:
                features_0 = np.array([0,0,0])
                for module in species.members:
                    features = np.array(module.genotype_phenotype_features(self.historical_marker))
                    features_0 = np.add(features_0, features)
                module_species_average_features[species.id]["features"].append((features_0/len(species.members)).tolist())
                module_species_average_features[species.id]["generations"].append(-1)

        for species in self.blueprint_species_list:
            if species.id not in list(blueprint_species_average_features.keys()):
                blueprint_species_average_features[species.id] = {"generations":[],"features":[]}
            else:
                features_0 = np.array([0, 0, 0, 0, 0])
                for blueprint in species.members:
                    features = np.array(blueprint.genotype_phenotype_features(self.historical_marker))
                    features_0 = np.add(features_0, features)
                blueprint_species_average_features[species.id]["features"].append((features_0/len(species.members)).tolist())
                blueprint_species_average_features[species.id]["generations"].append(-1)

        for generation in range(evo_parameters["generations"]):
            t3 = time.perf_counter()
            logging.info("*" * 40)
            logging.info("Evolving for generation: " + str(generation))
            self.evolve_blueprints(evo_parameters,self.historical_marker, "modules")
            print("()" * 40)
            self.evolve_blueprints(evo_parameters, self.historical_marker, "blueprints")
            for blueprint in self.blueprints:
                pdfsummary.plot_blueprints(blueprint, "generation" + str(generation), show=True,root_path = self.root_path)# pdfpages=self.pdfpages)
            path_map = self.fit_blueprints(generation)
            fitness_blueprints = []
            for key, value in self.fitness_map_blueprints.items():
                fitness_blueprints.append(value[-1])
            pdfsummary.plot_generation_blueprint_fitnes(fitness_blueprints, pdfpages=self.pdfpages)
            self.fit_modules()
            self.fit_species()
            best_blueprint = self.get_best_blueprint()
            self.clean_up(path_map, best_blueprint)
            path = self.trainer.save_blueprint_model([best_blueprint], self.supernets_map, self.best_supernet, generation)
            saved_paths.append(path)
            saved_paths_original.append(path_map[best_blueprint.id])
            n_best_blueprint.append(best_blueprint.fitness())
            t4 = time.perf_counter()
            logging.info("Best blueprint id " + str(best_blueprint.id) + " with fitness " + str(
                best_blueprint.fitness()) + " for generation " + str(generation))
            logging.info("Time for generation: " + str(generation) + " --- " + str(t4 - t3))
            logging.info("Generation complete")
            logging.info("*" * 40)
            blueprin_species_dict[str(generation)] = {}
            module_species_dict[str(generation)] = {}
            for species in self.blueprint_species_list:
                blueprin_species_dict[str(generation)][str(species.id)] = len(species.members)
            for species in self.module_species_list:
                module_species_dict[str(generation)][str(species.id)] = len(species.members)


            for species in self.module_species_list:
                if species.id not in list(module_species_average_features.keys()):
                    module_species_average_features[species.id] = {"generations":[],"features":[]}
                else:
                    features_0 = np.array([0, 0, 0])
                    for module in species.members:
                        features = np.array(module.genotype_phenotype_features(self.historical_marker))
                        features_0 = np.add(features_0, features)
                    module_species_average_features[species.id]["features"].append((features_0 / len(species.members)).tolist())
                    module_species_average_features[species.id]["generations"].append(generation)

            for species in self.blueprint_species_list:
                if species.id not in list(blueprint_species_average_features.keys()):
                    blueprint_species_average_features[species.id] = {"generations":[],"features":[]}
                else:
                    features_0 = np.array([0, 0, 0, 0, 0])
                    for blueprint in species.members:
                        features = np.array(blueprint.genotype_phenotype_features(self.historical_marker))
                        features_0 = np.add(features_0, features)
                    blueprint_species_average_features[species.id]["features"].append((features_0 / len(species.members)).tolist())
                    blueprint_species_average_features[species.id]["generations"].append(generation)


        pdfsummary.plot_average_species_features(module_species_average_features,"modules",pdfpages=self.pdfpages)
        pdfsummary.plot_average_species_features(blueprint_species_average_features, "blueprints", pdfpages=self.pdfpages)
        pdfsummary.plot_generation_best_fitnes(n_best_blueprint, pdfpages=self.pdfpages)
        pdfsummary.plot_species_count(blueprin_species_dict,pdfpages=self.pdfpages)

        t5 = time.perf_counter()
        logging.info("Time for intialization --- " + str(t2 - t1))
        logging.info("Time for Evolution --- " + str(t5 - t1))
        logging.info("Improvement of fitness over generations")
        logging.info(n_best_blueprint)
     

        validation_results_original = []
        for saved_path_original in saved_paths_original:
            results = self.trainer.validate_test(saved_path_original)
            validation_results_original.append(results[1])
        logging.info("saved at the original point")
        logging.info(validation_results_original)
        pdfsummary.plot_generation_best_fitnes(validation_results_original, pdfpages=self.pdfpages)

        validation_results = []
        for saved_path in saved_paths:
            results = self.trainer.validate_test(saved_path)
            validation_results.append(results[1])
        logging.info("Saved at the end of generation")
        logging.info(validation_results)
        pdfsummary.plot_generation_best_fitnes(validation_results, pdfpages=self.pdfpages)






        return n_best_blueprint

    def initialization(self, evo_parameters):
        layers, additional_layers, sequential_components, supernet_config = self.config_provider.initialize_data()
        initialization = Initialization(layers, additional_layers, random_complements=True,
                                        supernet_layers=supernet_config)
        supernets = initialization.initialize_supernet_from_components(evo_parameters["supernet"]["population_size"],
                                                                       evo_parameters["supernet"][
                                                                           "minimal_individual_sizes"])
        modules = initialization.initialize_modules_from_supernet(supernets[0],
                                                                  evo_parameters["modules"]["population_size"],
                                                                  self.historical_marker)
        blueprints = initialization.initialize_blueprints(evo_parameters["blueprints"]["population_size"],
                                                          evo_parameters["blueprints"]["minimal_individual_sizes"],
                                                          modules,
                                                          self.historical_marker, supernets[0], self.supernets_map)
        self.modules = modules
        self.create_species(modules, "modules")
        self.blueprints = blueprints
        self.create_species(blueprints, "blueprints")
        self.supernets = supernets
        self.best_supernet = supernets[0]

    def clean_up(self,path_map,blueprint):
        for key, value in path_map.items():
            if key != blueprint.id:
                logging.info("Cleaning up saved model at path " + value)
                shutil.rmtree(value)
            if key == blueprint.id:
                logging.info("Best blueprint at " + value)
    def tournament_selection(self, tournament_size, num_tournaments, population):

        selected_individuals = []
        if len(population) >= 2:
            for i in range(num_tournaments):
                participants = []
                participants_fitness = []
                for j in range(tournament_size):
                    participant = choice(population)
                    participants.append(participant)
                    participants_fitness.append(participant.fitness())

                index = np.argmax(np.array(participants_fitness))
                selected_individuals.append(participants[int(index)])
        else:
            selected_individuals = population * num_tournaments

        return selected_individuals

    def create_species(self, individuals, individual_type):

        if individual_type not in ["modules", "blueprints"]:
            raise ValueError("type must be one of blueprints or modules")

        members = []
        generated_species = []
        for i in range(len(individuals)):
            members.append(individuals[i])
            individuals[i].species = 0

        species_new = Species(0, members, individual_type, "Species_" + individual_type + "_" + str(0))
        generated_species.append(species_new)

        if individual_type == "blueprints":
            self.blueprint_species_list = generated_species
        elif individual_type == "modules":
            self.module_species_list = generated_species

    def update_species(self, new_individuals, historical_marker, individual_type, delta=0.0005):

        if individual_type == "modules":
            species_list = self.module_species_list
            individuals = self.modules
        elif individual_type == "blueprints":
            species_list = self.blueprint_species_list
            individuals = self.blueprints
        else:
            raise ValueError("type must be one of blueprints or modules")

        create_new_species = False
        old_features = [individual.genotype_phenotype_features(historical_marker) for individual in
                                       individuals]
        old_labels = [individual.species for individual in individuals]

        new_features = [new_individual.genotype_phenotype_features(historical_marker) for new_individual in
                        new_individuals]
        scaled_features = scale(old_features + new_features)

        old_features = scaled_features[:len(old_features)]
        new_features = scaled_features[len(old_features):]

        print("Old Labels", individual_type)
        print(old_labels)
        if len(set(old_labels)) < 2:
            dominant_species_id = 0
            for species in species_list:
                if species.members:
                    dominant_species_id = species.id

            kmeans = KMeans(n_clusters=1, random_state=0)
            kmeans.fit(old_features)
            centroids = kmeans.cluster_centers_

            cluster_distances_map = {}

            unique_labels = list(set(old_labels))

            unique_labels.sort()
            centroid_map = {}
            for i in range(len(unique_labels)):
                centroid_map[unique_labels[i]] = centroids[i]

            for i in unique_labels:
                cluster_distances_map[i] = []
            for i in range(len(old_features)):
                cluster_distances_map[old_labels[i]].append(
                    np.sum(pairwise_distances(old_features[i].reshape(1, -1),
                                              np.array(centroid_map[old_labels[i]]).reshape(1, -1), force_all_finite = True)))

            max_point_distance_clusters = {}

            for cluster, distances in cluster_distances_map.items():
                if np.max(distances) == 0:
                    max_point_distance_clusters[cluster] = 1
                else:
                    max_point_distance_clusters[cluster] = np.max(distances)

            new_labels = kmeans.predict(new_features)
            dominant_labels = [dominant_species_id] * len(new_features)
            print(dominant_labels)
            adjusted_labels = []

            for i in range(len(new_features)):
                print(i)
                d1 = np.sum(pairwise_distances(new_features[i].reshape(1, -1),
                                               np.array(centroid_map[new_labels[i]]).reshape(1, -1), force_all_finite = True))
                print("Print Distance")
                print( max_point_distance_clusters[new_labels[i]])
                print(((d1 - max_point_distance_clusters[new_labels[i]]) / max_point_distance_clusters[new_labels[i]]))
                print(delta)
                if ((d1 - max_point_distance_clusters[new_labels[i]]) / max_point_distance_clusters[new_labels[i]]) > delta:
                    adjusted_labels.append(unique_labels[-1] + 1)
                    print("Found unique individual")
                    create_new_species = True
                else:
                    adjusted_labels.append(dominant_labels[i])

            if create_new_species:
                species_list.append(
                    Species(unique_labels[-1] + 1, [], individual_type,
                            "Species_" + individual_type + "_" + str(unique_labels[-1] + 1)))
                print("Created new Species")

            for i in range(len(new_individuals)):
                new_individuals[i].species = adjusted_labels[i]

            species_exclusion_list = []

            for species in species_list:
                members = []
                for new_individual in new_individuals:
                    if new_individual.species == species.id:
                        members.append(new_individual)
                if members:
                    species.members = species.members + members
                if not species.members:
                    species_exclusion_list.append(species)

            for species in species_exclusion_list:
                species_list.remove(species)
        else:
            if not new_individuals:
                raise Exception("No new offsprings")

            NCcLf = NearestCentroid(metric="euclidean")
            NCcLf.fit(old_features, old_labels)
            centroids = NCcLf.centroids_
            print("centroids")
            print(NCcLf.centroids_)

            cluster_distances_map = {}


            unique_labels = list(set(old_labels))

            unique_labels.sort()
            centroid_map = {}
            for i in range(len(unique_labels)):
                centroid_map[unique_labels[i]] = centroids[i]


            for i in unique_labels:
                cluster_distances_map[i] = []
            for i in range(len(old_features)):
                cluster_distances_map[old_labels[i]].append(
                    np.sum(pairwise_distances(old_features[i].reshape(1,-1), np.array(centroid_map[old_labels[i]]).reshape(1,-1),force_all_finite = True)))

            max_point_distance_clusters = {}

            for cluster, distances in cluster_distances_map.items():
                if np.max(distances) == 0:
                    max_point_distance_clusters[cluster] = 1
                else:
                    max_point_distance_clusters[cluster] = np.max(distances)

            new_labels = NCcLf.predict(new_features)
            adjusted_labels = []

            for i in range(len(new_features)):
                d1 = np.sum(pairwise_distances(new_features[i].reshape(1,-1), np.array(centroid_map[new_labels[i]]).reshape(1,-1),force_all_finite = True))
                print("Print Distance")
                print(max_point_distance_clusters[new_labels[i]])
                print(((d1 - max_point_distance_clusters[new_labels[i]]) / max_point_distance_clusters[new_labels[i]]))
                if ((d1 - max_point_distance_clusters[new_labels[i]]) / max_point_distance_clusters[new_labels[i]]) > delta:
                    adjusted_labels.append(unique_labels[-1] + 1)
                    create_new_species = True
                else:
                    adjusted_labels.append(new_labels[i])
            non_existing = True
            if create_new_species:
                for species in species_list:
                    if species.id == unique_labels[-1] + 1:
                        raise Exception("Creating Duplicate species")
                species_list.append(
                    Species(unique_labels[-1] + 1, [], individual_type, "Species_" + individual_type + "_" + str(unique_labels[-1] + 1)))
                print("Created new Species")

            for i in range(len(new_individuals)):
                new_individuals[i].species = adjusted_labels[i]

            species_exclusion_list = []

            for species in species_list:
                members = []
                for new_individual in new_individuals:
                    if new_individual.species == species.id:
                        members.append(new_individual)
                if members:
                    species.members = species.members + members
                if not species.members:
                    species_exclusion_list.append(species)

            for species in species_exclusion_list:
                species_list.remove(species)

    def fit_species(self):
        for species in self.module_species_list + self.blueprint_species_list:
            species.update_fitness()

    def evolve_blueprints(self, evo_parameters, historical_marker, individual_type):
        if individual_type not in ["modules", "blueprints"]:
            print("Individual type")
            raise ValueError("type must be one of blueprints or modules")

        offsprings, exclusions = self.crossover_blueprints(individual_type, evo_parameters[individual_type]["generation_gap"],
                                                           evo_parameters[individual_type]["crossover_probability"])
        self.mutate_blueprints(offsprings, individual_type, evo_parameters[individual_type]["mutation_rate"])
        self.update_species(offsprings, historical_marker,individual_type, evo_parameters[individual_type]["delta"])
        self.replace_blueprints(offsprings, exclusions, individual_type)

    def crossover_blueprints(self, individual_type, generation_gap=0.3, crossover_probability=0.99):

        if individual_type == "modules":
            crossover_individuals = self.modules
            crossover_species_list = self.module_species_list
        elif individual_type == "blueprints":
            crossover_individuals = self.blueprints
            crossover_species_list = self.blueprint_species_list
        else:
            raise ValueError("type must be one of blueprints or modules")

        lambda_offsprings = int(round(len(crossover_individuals) * generation_gap))
        species_cumulative_fitness = 0
        for species in crossover_species_list:
            species_cumulative_fitness = species_cumulative_fitness + (species.fitness() if (len(species.members) >= 1)
                                                                       else 0)

        offspring_count_map = {}
        total_count = 0
        crossover_species_list.sort(key=lambda x: x.fitness(), reverse=True)
        for species in crossover_species_list:
            if len(species.members) >= 1:
                if species_cumulative_fitness != 0:
                    proportion = (species.fitness() / species_cumulative_fitness)
                else:
                    proportion = 1 / len(crossover_species_list)

                proportional_count = int(max(1, round(lambda_offsprings * proportion)))
                total_count = int(total_count + proportional_count)
                if total_count <= lambda_offsprings:
                    offspring_count_map[species] = proportional_count
                else:
                    offspring_count_map[species] = lambda_offsprings - (total_count - proportional_count)


        exclusions = []
        offsprings = []
        for species, offspring_count in offspring_count_map.items():
            species.members.sort(key=lambda x: x.fitness(), reverse=True)
            if offspring_count > 0:

                print( individual_type + " Offspring count: " + str(offspring_count))
                if len(species.members) > offspring_count:
                    exclusions = exclusions + species.members[-offspring_count:]
                    parents = self.tournament_selection(2, int(2 * offspring_count), species.members)
                else:
                    exclusions = exclusions + species.members[-1:]
                    parents = self.tournament_selection(2, 2, species.members)

                if len(exclusions) != len(list(set(exclusions))):
                    print("Found duplicates")


                while (len(parents) != 0):
                    print("While loop  crossover")
                    random_point = uniform(0, 1)
                    if random_point < crossover_probability:
                        parent1 = choice(parents)
                        parents.remove(parent1)
                        if len(parents) >= 1:
                            parent2 = choice(parents)
                            parents.remove(parent2)
                        else:
                            parent2 = parent1
                    else:
                        parent1 = choice(parents)
                        parents.remove(parent1)
                        parent2 = choice(parents)
                        parents.remove(parent2)
                        parent1 = choice([parent1, parent2])
                        parent2 = parent1
                    child = parent1.crossover(parent2, self.historical_marker)
                    offsprings.append(child)
        print(individual_type + " Offsprings " + str(len(offsprings)))
        if len(offsprings) != len(exclusions):
            print("Mismatch in offsprings count in ", individual_type)

        return offsprings, exclusions

    def mutate_blueprints(self, individuals, individual_type, mutation_rate=0.5):

        if individual_type not in ["modules", "blueprints"]:
            raise ValueError("type must be one of blueprints or modules")

        for individual in individuals:
            random_point = uniform(0, 1)
            if random_point < mutation_rate:
                if individual_type == "modules":
                    individual.mutate()
                if individual_type == "blueprints":
                    individual.mutate(self.modules, self.historical_marker, self.best_supernet, self.supernets_map)

    def replace_blueprints(self, offsprings, exclusions, individual_type, elitism_rate=None):

        if individual_type == "modules":
            individuals = self.modules
            species_list = self.module_species_list
        elif individual_type == "blueprints":
            individuals = self.blueprints
            species_list = self.blueprint_species_list
        else:
            raise ValueError("type must be one of blueprints or modules")



        individuals.sort(key=lambda x: x.fitness(), reverse=True)
        final_exclusion = []
        if elitism_rate is not None:
            elite_blueprints_count = int(len(individuals) * elitism_rate)
            for i in range(len(offsprings)):
                random_non_elite_blueprint = choice(individuals[elite_blueprints_count:])
                while(random_non_elite_blueprint in final_exclusion):
                    print("Stuck in while loop")
                    random_non_elite_blueprint = choice(individuals[elite_blueprints_count:])

                final_exclusion.append(random_non_elite_blueprint)
        else:
            final_exclusion = exclusions

        species_exclusion_list = []
        print("Length of offsprings vs final exclusions" , len(offsprings), len(final_exclusion))
        for offspring, reject_offspring in zip(offsprings, final_exclusion):
            if reject_offspring in individuals:
                individuals.remove(reject_offspring)
            else:
                print(reject_offspring)
                raise Exception("Invalid offspring")
            for species in species_list:
                if species.id == reject_offspring.species:
                    species.members.remove(reject_offspring)
                    if not species.members:
                        species_exclusion_list.append(species)
            #odd case of reject offspring not present in list
            individuals.append(offspring)

        for species in species_exclusion_list:
            species_list.remove(species)
        print("Indviduals count: ", len(individuals))
        for species in species_list:
            print("species count:", len(species.members))

    def fit_blueprints(self,generation):
        self.training_map_blueprints, self.fitness_map_blueprints, path_map = self.trainer.train_supernets_by_blueprints(
            self.blueprints, self.supernets_map, generation, save=True)

        for blueprint in self.blueprints:
            blueprint.update_fitness(self.fitness_map_blueprints[blueprint.id])

        return path_map

    def get_best_blueprint(self):
        self.blueprints.sort(key=lambda x: x.fitness(), reverse=True)
        return self.blueprints[0]

 
    def fit_modules(self):

        for blueprint in self.blueprints:
            for node in nx.algorithms.topological_sort(blueprint.graph):
                module = blueprint.graph.nodes[node]["module"]
                module.update_cooperative_fitness_values(blueprint.fitness_values)

        for species in self.module_species_list:
            size = len(species.members)
            for member in species.members:
                member.update_shared_fitness(size)

    def get_best_module(self):
        self.modules.sort(key=lambda x: x.fitness(), reverse=True)
        return self.modules[0]

