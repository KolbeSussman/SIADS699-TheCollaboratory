'''
Created by Kolbe Sussman & Qunkun Ma
Uses raw datset from OpenAlex and does the following:
- extracts only the needed columns
- extacts needed authorship information from json text in "authorship" column
- extracts topic information from json text in "topics" columns
'''

import os
import pandas as pd
import astx

# Paths
RAW_CSV = "/Users/kolbesussman/Downloads/umich_works_selected_columns.csv"
PROCESSED_CSV = "data/processed/umich_works_cleaned.csv"

## Ensure output directory exists
os.makedirs(os.path.dirname(PROCESSED_CSV), exist_ok=True)

# Load Data
print("Loading raw data...")
df = pd.read_csv(RAW_CSV)
print(f"Loaded {len(df)} rows")

# Drop duplicates based on 'id' or 'doi'
df = df.drop_duplicates(subset=['id', 'doi'])

# drop unneeded columns
#### QUNKUN HAS THIS CODE SOEMWHERE


# Extract authorship info
df["authorships_parsed"] = df["authorships"].apply(ast.literal_eval)

## list of author IDs
def extract_author_ids(authorships):
    return [
        author["author"]["id"]
        for author in authorships
        if author.get("author")
    ]

df["author_ids"] = df["authorships_parsed"].apply(extract_author_ids)


## list of author names
def extract_author_names(authorships):
    return [
        author["author"]["display_name"]
        for author in authorships
        if author.get("author")
    ]

df["author_names"] = df["authorships_parsed"].apply(extract_author_names)


## list of author institutions
def extract_institutions(authorships):
    institutions = []
    
    for author in authorships:
        for inst in author.get("institutions", []):
            name = inst.get("display_name")
            if name:
                institutions.append(name)
                
    return list(set(institutions))

df["institutions"] = df["authorships_parsed"].apply(extract_institutions)


## list of author affiliations - will be department/unit if available, otherwise will be U-M
def extract_raw_affiliations(authorships):
    affiliations = []

    for author in authorships:
        for aff in author.get("affiliations", []):
            raw = aff.get("raw_affiliation_string")
            if raw:
                dept = raw.split(",")[0].strip()
                affiliations.append(dept)

    return list(set(affiliations))

df["raw_affiliations"] = df["authorships_parsed"].apply(extract_raw_affiliations)

# Extract Topics
df["topics_parsed"] = df["topics"].apply(ast.literal_eval)

## extract topic ID
def extract_topic_ids(topics):
    return [t.get("id") for t in topics if t.get("id")]

df["topic_ids"] = df["topics_parsed"].apply(extract_topic_ids)

## extract topic display_name
def extract_display_name(topics):
    return [t.get("display_name") for t in topics if t.get("display_name")]

df["display_names"] = df["topics_parsed"].apply(extract_display_name)

## extract topic score
def extract_topic_scores(topics):
    return [t.get("score") for t in topics if t.get("score") is not None]

df["topic_scores"] = df["topics_parsed"].apply(extract_topic_scores)