import load_graph
import weight
import genetic_algorithm

consumtion, G_pos = load_graph.load_graph_from_csv("data.csv", truncate_at=100000)
print(consumtion, G_pos)


ga = genetic_algorithm.GraphGeneticAlgorithm(G_pos, consumtion, alpha=1, beta=1, pop_size=50, n_gen=50)
best_ind, best_fit = ga.run()
print("Best individual:", best_ind)
print("Fitness:", best_fit)