import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

"""
This script constructs a directed disaster-chain network,
visualizes the network, computes centrality metrics,
and exports node- and edge-level results.
Raw data are not included due to confidentiality restrictions.
"""

# 1. Load data (update paths as needed)
disaster_chain_df = pd.read_excel(
    r"file.xlsx"
)
disaster_type_df = pd.read_excel(
    r"file.xlsx"
)

# 2. Map disaster types to numeric IDs
disaster_type_dict = dict(
    zip(
        disaster_type_df["disaster_type"],
        disaster_type_df["disaster_type_id"]
    )
)

# 3. Replace disaster types with numeric IDs in the chain table
disaster_chain_mapped = disaster_chain_df.copy()
for col in disaster_chain_df.columns[3:]:
    disaster_chain_mapped[col] = disaster_chain_df[col].map(disaster_type_dict)

# 4. Build a directed network
G = nx.DiGraph()

for _, row in disaster_chain_mapped.iterrows():
    previous_node = None
    for disaster_type in row[3:]:
        if pd.notna(disaster_type):
            G.add_node(disaster_type)
            if previous_node is not None:
                G.add_edge(previous_node, disaster_type)
            previous_node = disaster_type

# 5. Circular layout for visualization
nodes = list(G.nodes())
radius = 2.0
node_count = len(nodes)

pos = {}
for i, node in enumerate(nodes):
    angle = 2 * np.pi * i / node_count
    pos[node] = (
        radius * np.cos(angle),
        radius * np.sin(angle)
    )

# 6. Plot directed network
plt.figure(figsize=(10, 10))

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=300,
    node_color="skyblue"
)

nx.draw_networkx_edges(
    G,
    pos,
    edgelist=G.edges(),
    edge_color="gray",
    width=2,
    arrowstyle="-|>",
    arrowsize=20
)

labels = {node: str(int(node)) for node in G.nodes()}
nx.draw_networkx_labels(
    G,
    pos,
    labels=labels,
    font_size=14,
    font_color="black",
    font_family="Times New Roman"
)

plt.axis("off")
plt.savefig(
    r"file.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# 7. Network centrality metrics
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
edge_betweenness_centrality = nx.edge_betweenness_centrality(G)

# 8. Node vulnerability (based on betweenness centrality)
node_vulnerability = {
    node: betweenness_centrality[node]
    for node in G.nodes()
}

# 9. Export results to Excel
node_results = {
    "Node": list(G.nodes()),
    "Degree Centrality": [degree_centrality.get(n, 0) for n in G.nodes()],
    "Betweenness Centrality": [betweenness_centrality.get(n, 0) for n in G.nodes()],
    "Closeness Centrality": [closeness_centrality.get(n, 0) for n in G.nodes()],
    "Eigenvector Centrality": [eigenvector_centrality.get(n, 0) for n in G.nodes()],
    "Vulnerability (Betweenness)": [node_vulnerability.get(n, 0) for n in G.nodes()]
}

edge_results = {
    "Edge": list(G.edges()),
    "Edge Betweenness Centrality": [
        edge_betweenness_centrality.get(e, 0)
        for e in G.edges()
    ]
}

node_df = pd.DataFrame(node_results)
edge_df = pd.DataFrame(edge_results)

with pd.ExcelWriter(
    r"C:\Users\AL\Desktop\file.xlsx"
) as writer:
    node_df.to_excel(writer, sheet_name="Node Centrality", index=False)
    edge_df.to_excel(writer, sheet_name="Edge Betweenness", index=False)

print("Network analysis results have been exported successfully.")
