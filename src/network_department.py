"""
Created by: Kolbe Sussman

Creates networks of coauthorship at the department level
using pre-parsed raw_affiliations column
"""

import pandas as pd
import networkx as nx
from itertools import combinations
from collections import Counter
import ast

# Load cleaned data
DATA_PATH = "../data/processed/umich_works_cleaned.csv"
df = pd.read_csv(DATA_PATH)

# Extract department-level edges using pre-parsed raw_affiliations
edges = []

for _, row in df.iterrows():
    depts = row['raw_affiliations']  # already a list of department strings
    
    # If it's stored as a string (from CSV), convert back to list
    if isinstance(depts, str):
        depts = ast.literal_eval(depts)
    
    depts = list(set(depts))  # unique departments per paper

    # Skip papers with fewer than 2 departments
    if len(depts) < 2:
        continue

    # Create all pairs of departments for this paper
    for dept_pair in combinations(depts, 2):
        edges.append(tuple(sorted(dept_pair)))  # sort to avoid duplicates

# Count weights
edge_weights = Counter(edges)

# Build NetworkX graph
G = nx.Graph()
for (dept1, dept2), weight in edge_weights.items():
    G.add_edge(dept1, dept2, weight=weight)

# Compute metrics
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight')

# Save network metrics
metrics_df = pd.DataFrame({
    'department': list(G.nodes),
    'degree_centrality': [degree_centrality[d] for d in G.nodes],
    'betweenness_centrality': [betweenness_centrality[d] for d in G.nodes],
    'eigenvector_centrality': [eigenvector_centrality[d] for d in G.nodes]
})

metrics_df.to_csv("../data/processed/department_network_metrics.csv", index=False)

# Save full edge list
edges_df = pd.DataFrame([
    {'department_1': d1, 'department_2': d2, 'weight': w}
    for (d1, d2), w in edge_weights.items()
])
edges_df.to_csv("../data/processed/department_network_edges.csv", index=False)

print("Department network created and metrics saved!")