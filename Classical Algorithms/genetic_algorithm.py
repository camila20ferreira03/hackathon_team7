import networkx as nx
import random
from deap import base, creator, tools
import weight
import load_graph
import matplotlib.pyplot as plt

class GraphGeneticAlgorithm:
    def __init__(self, graph, consumption=0, alpha=1, beta=1,
                 pop_size=50, n_gen=100, cx_prob=0.7, mut_prob=0.2, mut_indpb=0.05):
        self.G = graph
        self.n_nodes = self.G.number_of_nodes()
        self.ALPHA = alpha
        self.BETA = beta
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.mut_indpb = mut_indpb
        self.consumption = consumption
        self.FITNESS = []

        # DEAP setup
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_bool, n=self.n_nodes)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxOnePoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=self.mut_indpb)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.fitness_function)

    def individual_to_graph(self, individual):
        active_nodes = [list(self.G.nodes())[i] for i, b in enumerate(individual) if b == 1]
        sub_graph = self.G.subgraph(active_nodes)
        return sub_graph

    def fitness_function(self, individual):
        subgraph = self.individual_to_graph(individual)
        if subgraph.number_of_edges() == 0:
            return (float('inf'),)
        weight_sum = sum(weight.weight_node(n) for n in subgraph.nodes(data=True))
        edges_count = subgraph.number_of_edges()
        return (abs(self.ALPHA*(1/edges_count)) + abs(self.BETA*(weight_sum + self.consumption)),)

    def run(self):
        population = self.toolbox.population(n=self.pop_size)
        for gen in range(self.n_gen):
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            
            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cx_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutation
            for mutant in offspring:
                if random.random() < self.mut_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            best = tools.selBest(population, 1)[0]
            population[:] = offspring
            population[0] = best
            if not best.fitness.valid:
                best.fitness.values = self.toolbox.evaluate(best)
            fits = [ind.fitness.values[0] for ind in population]
            print(f"Generation {gen}: Best fitness = {min(fits)}")
            self.FITNESS.append(min(fits))

        # Best individual
        best_ind = tools.selBest(population, 1)[0]
        return self.FITNESS, best_ind, best_ind.fitness.values[0]