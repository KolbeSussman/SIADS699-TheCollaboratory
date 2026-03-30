"""
Created by: Kolbe Sussman

Creates networks of coauthorship at the author level
using pre-parsed authorship columns
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

print(df.columns.tolist())

df['author_names'] = df['author_names'].apply(safe_parse)
df['author_ids'] = df['author_ids'].apply(safe_parse)


# Extract author-level edges using pre-parsed column
MAX_AUTHORS = 20  # skip papers with too many authors

edge_weights = Counter()
author_info = {}

for row in df.itertuples(index=False):
    author_names = row.author_names
    author_ids = row.author_ids

    if not author_names or len(author_names) < 2:
        continue

    if len(author_names) > MAX_AUTHORS:
        continue


    # store metadata
    for i, name in enumerate(author_names):
        if name not in author_info:
            author_info[name] = {
                'id': author_ids[i] if i < len(author_ids) else None
            }

    # generate edges
    for a, b in combinations(sorted(author_names), 2):
        edge_weights[(a, b)] += 1

    G = nx.Graph()

for i, ((auth1, auth2), weight) in enumerate(edge_weights.items(), start=1):
    G.add_edge(auth1, auth2, weight=weight)

# Compute metrics
degree_centrality = nx.degree_centrality(G)
#betweenness_centrality = nx.betweenness_centrality(G, weight='weight') # This takes HUGE computing power that I simply do not have
eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight')

# Save network metrics
metrics_df = pd.DataFrame({
    'author': list(G.nodes),
    'degree_centrality': [degree_centrality[a] for a in G.nodes],
    #'betweenness_centrality': [betweenness_centrality[a] for a in G.nodes],
    'eigenvector_centrality': [eigenvector_centrality[a] for a in G.nodes],
    'id': [author_info[a]['id'] for a in G.nodes]
})

metrics_df.to_csv("data/processed/author_network_metrics.csv", index=False)

# Save full edge list
edges_df = pd.DataFrame([
    {'author_1': a1, 'author_2': a2, 'weight': w}
    for (a1, a2), w in edge_weights.items()
])
edges_df.to_csv("data/processed/author_network_edges.csv", index=False)


print("Author network created and metrics saved!")