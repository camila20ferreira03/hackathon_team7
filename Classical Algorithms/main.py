import load_graph
import weight
import genetic_algorithm
import matplotlib.pyplot as plt

consumtion, G_pos = load_graph.load_graph_from_csv("simplified_dataset.csv", truncate_at=-1)
print(consumtion, G_pos)


ga = genetic_algorithm.GraphGeneticAlgorithm(G_pos, consumtion, alpha=1, beta=1000, pop_size=50, n_gen=100)
fitness, best_ind, best_fit = ga.run()
print("Best individual:", best_ind)
print("Fitness:", best_fit)
plt.plot(fitness)
plt.show()

active_nodes = [list(G_pos.nodes())[i] for i, b in enumerate(best_ind) if b == 1]
sub_graph = G_pos.subgraph(active_nodes)
sum = 0
for n in sub_graph.nodes(data=True):
    sum += weight.weight_node(n)
print("Weight sum of best individual:", sum)