"""
Created by: Kolbe Sussman

Creates networks of topic co-occurrence
using display_names column
"""

import pandas as pd
import networkx as nx
from itertools import combinations
from collections import Counter
import ast

df = pd.read_csv("data/processed/umich_works_cleaned.csv")

def safe_parse(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return []
    return []


# Parse topics
df['display_names'] = df['display_names'].apply(safe_parse)

MAX_TOPICS = 50  # topics are usually fewer → can allow more

edge_weights = Counter()

for row in df.itertuples(index=False):
    topics = row.display_names

    if not topics or len(topics) < 2:
        continue

    if len(topics) > MAX_TOPICS:
        continue

    # 🔑 Deduplicate topics within a paper
    topics = list(set([t for t in topics if t]))

    if len(topics) < 2:
        continue

    # generate edges
    for t1, t2 in combinations(sorted(topics), 2):
        edge_weights[(t1, t2)] += 1

# Build graph
G = nx.Graph()

for (t1, t2), weight in edge_weights.items():
    G.add_edge(t1, t2, weight=weight)

# Compute metrics
degree_centrality = nx.degree_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight')

# Save metrics
metrics_df = pd.DataFrame({
    'topic': list(G.nodes),
    'degree_centrality': [degree_centrality[t] for t in G.nodes],
    'eigenvector_centrality': [eigenvector_centrality[t] for t in G.nodes],
})

metrics_df.to_csv("data/processed/topic_network_metrics.csv", index=False)

# Save edges
edges_df = pd.DataFrame([
    {'topic_1': t1, 'topic_2': t2, 'weight': w}
    for (t1, t2), w in edge_weights.items()
])

edges_df.to_csv("data/processed/topic_network_edges.csv", index=False)

print("Topic network created and metrics saved!")