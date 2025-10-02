import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import weight

def load_graph_from_csv(csv_path = "data_reduced.csv", truncate_at=84):

    # 1. Load CSV
    df = pd.read_csv(csv_path, header=0, encoding="utf-8-sig", low_memory=False, sep=",")
    if truncate_at > 0:
        df = df.head(truncate_at)

    # 2. Build graph
    G_pos = nx.Graph()
    consumption = 0

    # --- Step B: Add prosumer nodes ---
    for idx, row in df.iterrows():
        if (weight.weight(row.to_dict()) > 0):
            prosumer_node = f"Prosumer_{idx}"
            G_pos.add_node(prosumer_node, **row.to_dict(), type="prosumer")
        else:
            consumption += weight.weight(row.to_dict())



    # --- Step C: Connect prosumers from different substations ---
    producers_nodes = [n for n, attr in G_pos.nodes(data=True) if attr["type"] == "prosumer"]

    for i, p1 in enumerate(producers_nodes):
        sub1 = G_pos.nodes[p1]["substation_id"]
        for p2 in producers_nodes[i+1:]:
            sub2 = G_pos.nodes[p2]["substation_id"]
            if sub1 != sub2:
                G_pos.add_edge(p1, p2)

    return consumption, G_pos


# 3. Draw graph (simple layout)
#plt.figure(figsize=(12, 8))

#pos = nx.spring_layout(G, seed=42)

# color by type
#colors = []
#for n, attr in G.nodes(data=True):
#    if attr["type"] == "substation":
#        colors.append("red")
#    else:
#        colors.append("blue")

#nx.draw(G, pos, with_labels=True, node_color=colors, node_size=500, font_size=8)
#plt.show()