'''
Created by Kolbe Sussman

'''

import os
import pandas as pd

# Paths
RAW_CSV = "../data/raw/umich_works_100k.csv"
PROCESSED_CSV = "../data/processed/umich_works_cleaned.csv"

## Ensure output directory exists
os.makedirs(os.path.dirname(PROCESSED_CSV), exist_ok=True)

# Load Data
print("Loading raw data...")
df = pd.read_csv(RAW_CSV)
print(f"Loaded {len(df)} rows")

# Basic Cleaning
## Drop duplicates based on 'id' or 'doi'
df = df.drop_duplicates(subset=['id', 'doi'])

## Fill missing values for key columns
df['display_name'] = df['display_name'].fillna('')
df['publication_year'] = df['publication_year'].fillna(0).astype(int)
df['primary_location'] = df['primary_location'].fillna('')
df['authorships'] = df['authorships'].fillna('[]')
df['institutions'] = df['institutions'].fillna('[]')
df['topics'] = df['topics'].fillna('[]')
df['concepts'] = df['concepts'].fillna('[]')

# Normalize Author Names
## Example: convert authorships from string to list of names
## Assuming 'authorships' column is JSON-like string from OpenAlex
import ast

def extract_author_names(authorship_str):
    try:
        authors_list = ast.literal_eval(authorship_str)
        names = [a.get('author', {}).get('display_name', '') for a in authors_list]
        return [n for n in names if n]
    except:
        return []

df['author_names'] = df['authorships'].apply(extract_author_names)

# Extract Departments / Institutions
def extract_departments(authorship_str):
    try:
        authors_list = ast.literal_eval(authorship_str)
        depts = []
        for a in authors_list:
            for inst in a.get('institutions', []):
                display_name = inst.get('display_name')
                if display_name:
                    depts.append(display_name)
        return list(set(depts))  # unique
    except:
        return []

df['departments'] = df['authorships'].apply(extract_departments)

# Extract Topics
def extract_topics(topics_str):
    try:
        topics_list = ast.literal_eval(topics_str)
        return [t.get('display_name') for t in topics_list if t.get('display_name')]
    except:
        return []

df['topics_cleaned'] = df['topics'].apply(extract_topics)

# Save Processed CSV
df.to_csv(PROCESSED_CSV, index=False)
print(f"Saved cleaned data to {PROCESSED_CSV}, {len(df)} rows")