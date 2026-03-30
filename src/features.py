"""
Created by: Kolbe Sussman

Generates ML-ready link prediction features for University of Michigan collaboration prediction.
Uses pre-built author, department, and topic networks plus publication metadata.
"""

import pandas as pd
import networkx as nx
from collections import defaultdict
from itertools import combinations
import ast
import random

# Helper Functions

def safe_parse(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return []
    return []

# Load Data

df = pd.read_csv("data/processed/umich_works_cleaned.csv")
df['author_ids'] = df['author_ids'].apply(safe_parse)
df['display_names'] = df['display_names'].apply(safe_parse)
df['raw_affiliations'] = df['raw_affiliations'].apply(safe_parse)

author_edges = pd.read_csv("data/processed/author_network_edges.csv")
author_metrics = pd.read_csv("data/processed/author_network_metrics.csv")

# Build Author Graph

G = nx.Graph()
for row in author_edges.itertuples(index=False):
    G.add_edge(row.author_1, row.author_2, weight=row.weight)

deg_dict = dict(zip(author_metrics['author'], author_metrics['degree_centrality']))
eig_dict = dict(zip(author_metrics['author'], author_metrics['eigenvector_centrality']))

# Build Author Metadata

author_topics = defaultdict(set)
author_affils = defaultdict(set)
author_papers = defaultdict(set)
author_citations = defaultdict(int)

for row in df.itertuples(index=False):
    for a in row.author_ids:
        if a:
            author_topics[a].update(row.display_names)
            author_affils[a].update(row.raw_affiliations)
            author_papers[a].add(row.id)
            author_citations[a] += row.cited_by_count

# Generate Candidate Pairs

# Positive examples: observed coauthor edges
positive_pairs = set(zip(author_edges['author_1'], author_edges['author_2']))

# Negative examples: random pairs without collaboration
all_authors = list(G.nodes)
negative_pairs = set()
while len(negative_pairs) < len(positive_pairs):
    a, b = random.sample(all_authors, 2)
    if (a, b) not in positive_pairs and (b, a) not in positive_pairs:
        negative_pairs.add((a, b))

# Feature Functions

def common_neighbors(a, b):
    try:
        return len(list(nx.common_neighbors(G, a, b)))
    except:
        return 0

def jaccard(a, b):
    try:
        return next(nx.jaccard_coefficient(G, [(a, b)]))[2]
    except:
        return 0

def degree_diff(a, b):
    return abs(deg_dict.get(a, 0) - deg_dict.get(b, 0))

def eigen_diff(a, b):
    return abs(eig_dict.get(a, 0) - eig_dict.get(b, 0))

def topic_overlap(a, b):
    t1 = author_topics[a]
    t2 = author_topics[b]
    if not t1 or not t2:
        return 0
    return len(t1 & t2) / len(t1 | t2)

def dept_overlap(a, b):
    d1 = author_affils[a]
    d2 = author_affils[b]
    if not d1 or not d2:
        return 0
    return len(d1 & d2) / len(d1 | d2)

def paper_diff(a, b):
    return abs(len(author_papers[a]) - len(author_papers[b]))

def citation_diff(a, b):
    return abs(author_citations[a] - author_citations[b])

# Build Feature Dataset

def build_row(a, b, label):
    return {
        "author_1": a,
        "author_2": b,
        "common_neighbors": common_neighbors(a, b),
        "jaccard": jaccard(a, b),
        "degree_diff": degree_diff(a, b),
        "eigen_diff": eigen_diff(a, b),
        "topic_overlap": topic_overlap(a, b),
        "dept_overlap": dept_overlap(a, b),
        "paper_diff": paper_diff(a, b),
        "citation_diff": citation_diff(a, b),
        "label": label
    }

rows = []

for a, b in positive_pairs:
    rows.append(build_row(a, b, 1))

for a, b in negative_pairs:
    rows.append(build_row(a, b, 0))

features_df = pd.DataFrame(rows)

# Save Features

features_df.to_csv("data/processed/link_prediction_features.csv", index=False)
print("Feature dataset created and saved! Rows:", len(features_df))