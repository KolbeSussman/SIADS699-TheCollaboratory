"""
Created by: Kolbe Sussman

Creates networks of coauthorship at the author level
using pre-parsed authorship columns
"""

import pandas as pd
import networkx as nx
from itertools import combinations
from collections import Counter

# Load cleaned data
DATA_PATH = "../data/processed/umich_works_cleaned.csv"
df = pd.read_csv(DATA_PATH)

# Extract author-level edges using pre-parsed column
edges = []
author_info = {}

for _, row in df.iterrows():
    author_names = row['author_names']
    raw_affiliations = row['raw_affiliations']
    author_ids = row['author_ids']

    # Sanity check: make sure lists exist
    if pd.isna(author_names) or len(author_names) < 2:
        continue  # skip papers with <2 authors - they do not connect to others

    # store metadata for each author
    for i, name in enumerate(author_names):
        if name not in author_info:
            author_info[name] = {
                'id': author_ids[i] if i < len(author_ids) else None,
                'affiliation': raw_affiliations[i] if i < len(raw_affiliations) else None
            }

    # generate all co-author pairs for this paper
    for pair in combinations(author_names, 2):
        edges.append(tuple(sorted(pair)))

# Count weights
edge_weights = Counter(edges)

# Build NetworkX graph
G = nx.Graph()
for (auth1, auth2), weight in edge_weights.items():
    G.add_edge(auth1, auth2, weight=weight)

# Compute metrics
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight')

# Save network metrics
metrics_df = pd.DataFrame({
    'author': list(G.nodes),
    'degree_centrality': [degree_centrality[a] for a in G.nodes],
    'betweenness_centrality': [betweenness_centrality[a] for a in G.nodes],
    'eigenvector_centrality': [eigenvector_centrality[a] for a in G.nodes],
    'id': [author_info[a]['id'] for a in G.nodes],
    'affiliation': [author_info[a]['affiliation'] for a in G.nodes]
})

metrics_df.to_csv("../data/processed/author_network_metrics.csv", index=False)

# Save full edge list
edges_df = pd.DataFrame([
    {'author_1': a1, 'author_2': a2, 'weight': w}
    for (a1, a2), w in edge_weights.items()
])
edges_df.to_csv("../data/processed/author_network_edges.csv", index=False)

print("Author network created and metrics saved!")