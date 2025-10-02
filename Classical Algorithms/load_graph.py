import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 1. Load CSV
df = pd.read_csv("data.csv", header=0, encoding="utf-8-sig", low_memory=False)

df = df.head(100)  # limit to first 20 rows for simplicity

# 2. Build graph
G = nx.Graph()

# --- Step A: Add substation nodes ---
substations = df["sub_station"].unique()
for s in substations:
    G.add_node(f"Substation_{s}", type="substation")

# --- Step B: Add prosumer nodes ---
for idx, row in df.iterrows():
    prosumer_node = f"Prosumer_{idx}"
    G.add_node(prosumer_node, **row.to_dict(), type="prosumer")
    # connect prosumer to its substation
    substation_node = f"Substation_{row['sub_station']}"
    G.add_edge(prosumer_node, substation_node)

# --- Step C: Connect prosumers from different substations ---
prosumer_nodes = [n for n, attr in G.nodes(data=True) if attr["type"] == "prosumer"]

for i, p1 in enumerate(prosumer_nodes):
    sub1 = G.nodes[p1]["sub_station"]
    for p2 in prosumer_nodes[i+1:]:
        sub2 = G.nodes[p2]["sub_station"]
        if sub1 != sub2:
            G.add_edge(p1, p2)

# 3. Draw graph (simple layout)
plt.figure(figsize=(12, 8))

pos = nx.spring_layout(G, seed=42)

# color by type
colors = []
for n, attr in G.nodes(data=True):
    if attr["type"] == "substation":
        colors.append("red")
    else:
        colors.append("blue")

nx.draw(G, pos, with_labels=True, node_color=colors, node_size=500, font_size=8)
plt.show()
