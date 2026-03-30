"""
Created by Kolbe Sussman & Qunkun Ma
Preprocessing script for OpenAlex U-M dataset:
- extracts needed columns
- extracts authorship info from JSON in "authorships"
- extracts topics info from JSON in "topics"
"""

import os
import pandas as pd
import ast

# Paths
RAW_CSV = "data/raw/umich_works_100k.csv"
PROCESSED_CSV = "data/processed/umich_works_cleaned.csv"

# Ensure output directory exists
os.makedirs(os.path.dirname(PROCESSED_CSV), exist_ok=True)

# Load data
print("Loading raw data...")
df = pd.read_csv(RAW_CSV)
print(f"Loaded {len(df)} rows")

# Drop duplicates based on 'id' or 'doi'
df = df.drop_duplicates(subset=['id', 'doi'])

# Keep only necessary columns
cols_to_keep = [
    'id', 'doi', 'title', 'authorships', 'topics',
    'primary_topic', 'cited_by_count', 'publication_year',
    'related_works', 'concepts'
]
df = df[cols_to_keep]

print("Parsing authorship information...")

def safe_literal_eval(x):
    """Safely parse JSON-like strings to Python objects."""
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return []
    elif isinstance(x, list):
        return x
    else:
        return []

df['authorships_parsed'] = df['authorships'].apply(safe_literal_eval)

def extract_author_ids(authorships):
    return [
        author.get("author", {}).get("id")
        for author in authorships
        if author.get("author") and author.get("author", {}).get("id")
    ]

def extract_author_names(authorships):
    return [
        author.get("author", {}).get("display_name")
        for author in authorships
        if author.get("author") and author.get("author", {}).get("display_name")
    ]

def extract_institutions(authorships):
    institutions = set()
    for author in authorships:
        for inst in author.get("institutions", []):
            name = inst.get("display_name")
            if name:
                institutions.add(name)
    return list(institutions)

def extract_raw_affiliations(authorships):
    affiliations = set()
    for author in authorships:
        for aff in author.get("affiliations", []):
            raw = aff.get("raw_affiliation_string")
            if raw:
                dept = raw.split(",")[0].strip()
                if dept:
                    affiliations.add(dept)
    return list(affiliations)

df['author_ids'] = df['authorships_parsed'].apply(extract_author_ids)
df['author_names'] = df['authorships_parsed'].apply(extract_author_names)
df['institutions'] = df['authorships_parsed'].apply(extract_institutions)
df['raw_affiliations'] = df['authorships_parsed'].apply(extract_raw_affiliations)

print("Parsing topic information...")

df['topics_parsed'] = df['topics'].apply(safe_literal_eval)

def extract_topic_ids(topics):
    return [t.get("id") for t in topics if t.get("id")]

def extract_display_names(topics):
    return [t.get("display_name") for t in topics if t.get("display_name")]

def extract_topic_scores(topics):
    return [t.get("score") for t in topics if t.get("score") is not None]

df['topic_ids'] = df['topics_parsed'].apply(extract_topic_ids)
df['display_names'] = df['topics_parsed'].apply(extract_display_names)
df['topic_scores'] = df['topics_parsed'].apply(extract_topic_scores)

df = df.groupby('title').first().reset_index()

df.to_csv(PROCESSED_CSV, index=False)
print(f"Saved {len(df)} records to {PROCESSED_CSV}")
print(df.columns.tolist())
print("Data processing finished.")