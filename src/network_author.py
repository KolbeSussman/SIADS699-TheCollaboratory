import pandas as pd
import ast
import itertools
from collections import Counter
import os

# Paths
INPUT_CSV = "data/processed/umich_works_cleaned.csv"
OUTPUT_CSV = "data/processed/author_edges.csv"

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Load cleaned data
print("Loading cleaned dataset...")
df = pd.read_csv(INPUT_CSV)

# Convert string lists back to Python lists
df["author_ids"] = df["author_ids"].apply(ast.literal_eval)

print(f"Papers loaded: {len(df)}")

# Build edge list
print("Generating co-authorship pairs...")

edge_counter = Counter()

for authors in df["author_ids"]:
    
    # Remove duplicates just in case
    authors = list(set(authors))
    
    # Generate all unique author pairs
    pairs = itertools.combinations(sorted(authors), 2)
    
    for pair in pairs:
        edge_counter[pair] += 1

print(f"Total unique edges: {len(edge_counter)}")

# Convert to dataframe
edges = []

for (author1, author2), weight in edge_counter.items():
    edges.append({
        "author1": author1,
        "author2": author2,
        "weight": weight
    })

edges_df = pd.DataFrame(edges)

# Save edge list
edges_df.to_csv(OUTPUT_CSV, index=False)

print(f"Edge list saved to {OUTPUT_CSV}")
print(f"Total edges: {len(edges_df)}")