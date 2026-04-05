"""
features.py (TEMPORAL VERSION)

Builds features for link prediction using:
- Past graph (<= cutoff year)
- Future collaborations as labels (> cutoff year)
"""

import pandas as pd
import networkx as nx
from itertools import combinations
import ast
import random
from tqdm import tqdm
import os

# Config
CUTOFF_YEAR = 2018

INPUT_CSV = "data/processed/umich_works_cleaned.csv"
OUTPUT_CSV = "data/processed/features_temporal.csv"

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Load + parse
df = pd.read_csv(INPUT_CSV)

for col in ['author_names', 'display_names', 'raw_affiliations']:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Split data
train_df = df[df['publication_year'] <= CUTOFF_YEAR]
test_df  = df[df['publication_year'] > CUTOFF_YEAR]

print(f"Train papers: {len(train_df)}")
print(f"Test papers: {len(test_df)}")

# Build TRAIN graph
G = nx.Graph()
MAX_AUTHORS = 20

for row in train_df.itertuples(index=False):
    authors = row.author_names
    if not authors or len(authors) < 2 or len(authors) > MAX_AUTHORS:
        continue
    authors = list(set(authors))
    for a, b in combinations(authors, 2):
        if G.has_edge(a, b):
            G[a][b]['weight'] += 1
        else:
            G.add_edge(a, b, weight=1)

print(f"Train graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

# Build FUTURE labels
future_edges = set()

for row in test_df.itertuples(index=False):
    authors = row.author_names
    if not authors or len(authors) < 2:
        continue
    authors = list(set(authors))
    for a, b in combinations(authors, 2):
        future_edges.add(tuple(sorted((a, b))))

print(f"Future edges: {len(future_edges)}")

# Candidate pairs (only authors seen in train)
authors = list(G.nodes)

# Positives = future collaborations
positive_pairs = [(a, b) for (a, b) in future_edges if a in G and b in G]

# Hard negative sampling
negatives = set()

while len(negatives) < len(positive_pairs):
    a, b = random.sample(authors, 2)
    pair = tuple(sorted((a, b)))

    if pair in future_edges:
        continue

    # HARD NEGATIVE: must share something in TRAIN graph
    neighbors_a = set(G.neighbors(a))
    neighbors_b = set(G.neighbors(b))

    if len(neighbors_a & neighbors_b) > 0:
        negatives.add(pair)

negative_pairs = list(negatives)

print(f"Positive: {len(positive_pairs)}, Negative: {len(negative_pairs)}")

# Author metadata (TRAIN ONLY)
train_exploded = train_df.explode('author_names')

author_papers = train_exploded.groupby('author_names')['id'].count().to_dict()
author_citations = train_exploded.groupby('author_names')['cited_by_count'].sum().to_dict()
author_topics = train_exploded.groupby('author_names')['display_names'].sum().apply(set).to_dict()
author_depts = train_exploded.groupby('author_names')['raw_affiliations'].sum().apply(set).to_dict()

# Feature functions
def jaccard(s1, s2):
    if not s1 or not s2:
        return 0
    return len(s1 & s2) / len(s1 | s2)


# Compute features
pairs = [(a,b,1) for a,b in positive_pairs] + [(a,b,0) for a,b in negative_pairs]

features = []

for a, b, label in tqdm(pairs, desc="Computing features"):
    neighbors_a = set(G.neighbors(a))
    neighbors_b = set(G.neighbors(b))

    features.append({
        'author_1': a,
        'author_2': b,
        'common_neighbors': len(neighbors_a & neighbors_b),
        'jaccard': jaccard(neighbors_a, neighbors_b),
        'degree_diff': abs(len(neighbors_a) - len(neighbors_b)),
        'topic_overlap': len(author_topics.get(a,set()) & author_topics.get(b,set())),
        'dept_overlap': len(author_depts.get(a,set()) & author_depts.get(b,set())),
        'paper_diff': abs(author_papers.get(a,0) - author_papers.get(b,0)),
        'citation_diff': abs(author_citations.get(a,0) - author_citations.get(b,0)),
        'label': label
    })

features_df = pd.DataFrame(features)
features_df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved {len(features_df)} rows to {OUTPUT_CSV}")