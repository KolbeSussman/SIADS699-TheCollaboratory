"""
Created by: Kolbe Sussman

Creates networks of coauthorship at the department level
using pre-parsed raw_affiliations column
"""
"""
Creates networks of co-affiliations (departments)
using raw_affiliations column
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

# Parse affiliations
df['raw_affiliations'] = df['raw_affiliations'].apply(safe_parse)

MAX_AFFILIATIONS = 20  # same idea as authors

edge_weights = Counter()

for row in df.itertuples(index=False):
    affiliations = row.raw_affiliations

    if not affiliations or len(affiliations) < 2:
        continue

    if len(affiliations) > MAX_AFFILIATIONS:
        continue

    # 🔑 IMPORTANT: deduplicate within paper
    affiliations = list(set([a for a in affiliations if a]))

    if len(affiliations) < 2:
        continue

    # generate edges
    for a, b in combinations(sorted(affiliations), 2):
        edge_weights[(a, b)] += 1

# Build graph (FIXED INDENTATION BUG)
G = nx.Graph()

for (aff1, aff2), weight in edge_weights.items():
    G.add_edge(aff1, aff2, weight=weight)

# Compute metrics
degree_centrality = nx.degree_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight')

# Save metrics
metrics_df = pd.DataFrame({
    'affiliation': list(G.nodes),
    'degree_centrality': [degree_centrality[a] for a in G.nodes],
    'eigenvector_centrality': [eigenvector_centrality[a] for a in G.nodes],
})

metrics_df.to_csv("data/processed/affiliation_network_metrics.csv", index=False)

# Save edges
edges_df = pd.DataFrame([
    {'affiliation_1': a1, 'affiliation_2': a2, 'weight': w}
    for (a1, a2), w in edge_weights.items()
])

edges_df.to_csv("data/processed/affiliation_network_edges.csv", index=False)

print("Affiliation network created and metrics saved!")