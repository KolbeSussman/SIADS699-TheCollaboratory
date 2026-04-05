"""
features.py
Created by: Kolbe Sussman & Qunkun Ma

Generates ML-ready features for link prediction based on precomputed networks:
- Author co-authorship network
- Department co-affiliation network
- Topic co-occurrence network

Features include graph-based, topic, department, paper, and citation metrics.
"""

import pandas as pd
import networkx as nx
from itertools import combinations
import ast
import os
from tqdm import tqdm
import random

# Paths
AUTHOR_METRICS = "data/processed/author_network_metrics.csv"
AUTHOR_EDGES = "data/processed/author_network_edges.csv"
AFFILIATION_METRICS = "data/processed/affiliation_network_metrics.csv"
TOPIC_METRICS = "data/processed/topic_network_metrics.csv"
WORKS_CSV = "data/processed/umich_works_cleaned.csv"
OUTPUT_CSV = "data/processed/features_pairs.csv"

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Load precomputed metrics
author_metrics = pd.read_csv(AUTHOR_METRICS).set_index('author')
affil_metrics = pd.read_csv(AFFILIATION_METRICS, engine='python', on_bad_lines='skip').set_index('affiliation')
topic_metrics = pd.read_csv(TOPIC_METRICS).set_index('topic')

# Build author graph from edges
edges_df = pd.read_csv(AUTHOR_EDGES)
G_author = nx.from_pandas_edgelist(edges_df, 'author_1', 'author_2', edge_attr='weight')

# Load works for metadata
df = pd.read_csv(WORKS_CSV)

# Parse list columns
list_cols = ['author_names', 'raw_affiliations', 'display_names']
for col in list_cols:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Aggregate per-author metadata
df_exploded = df.explode('author_names')

author_papers = df_exploded.groupby('author_names')['id'].count().to_dict()
author_citations = df_exploded.groupby('author_names')['cited_by_count'].sum().to_dict()
author_topics = df_exploded.groupby('author_names')['display_names'].sum().apply(set).to_dict()
author_depts = df_exploded.groupby('author_names')['raw_affiliations'].sum().apply(set).to_dict()

# Generate positive and negative pairs
positive_pairs = list(G_author.edges)

all_authors = list(G_author.nodes)
negative_pairs = set()
while len(negative_pairs) < len(positive_pairs):
    a1, a2 = random.sample(all_authors, 2)
    if not G_author.has_edge(a1, a2):
        negative_pairs.add((a1, a2))

pairs = [(a1, a2, 1) for a1, a2 in positive_pairs] + [(a1, a2, 0) for a1, a2 in negative_pairs]

# Helper function
def jaccard(set1, set2):
    if not set1 or not set2:
        return 0
    return len(set1 & set2) / len(set1 | set2)

# Compute features
features = []
for a1, a2, label in tqdm(pairs, desc="Computing features"):
    neighbors1, neighbors2 = set(G_author.neighbors(a1)), set(G_author.neighbors(a2))
    features.append({
        'author_1': a1,
        'author_2': a2,
        'common_neighbors': len(neighbors1 & neighbors2),
        'jaccard': jaccard(neighbors1, neighbors2),
        'degree_diff': abs(author_metrics.loc[a1, 'degree_centrality'] - author_metrics.loc[a2, 'degree_centrality']),
        'eigen_diff': abs(author_metrics.loc[a1, 'eigenvector_centrality'] - author_metrics.loc[a2, 'eigenvector_centrality']),
        'topic_overlap': len(author_topics.get(a1,set()) & author_topics.get(a2,set())),
        'dept_overlap': len(author_depts.get(a1,set()) & author_depts.get(a2,set())),
        'paper_diff': abs(author_papers.get(a1,0) - author_papers.get(a2,0)),
        'citation_diff': abs(author_citations.get(a1,0) - author_citations.get(a2,0)),
        'label': label
    })

# Save features
features_df = pd.DataFrame(features)
features_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(features_df)} feature rows to {OUTPUT_CSV}")