# SIADS699-TheCollaboratory
University of Michigan School of Information Masters of Applied Data Science Capstone project exploring networks of co-authorship and collaboration between scholars at U-M. 

# UofM Collaboration Prediction Project
## Overview

This project aims to predict potential academic collaborations among University of Michigan researchers using publication metadata from the OpenAlex API. The pipeline constructs multiple networks — author, department, and topic networks — and uses both network-based and publication-based features to train machine learning models for link prediction.

Predicted collaborations are probabilistic insights and are intended to highlight potential structural connections in research networks, not guaranteed collaborations.

## Project Structure

```
project_root/
│
├─ data/
│ ├─ raw/ # Raw OpenAlex JSON/CSV dumps
│ └─ processed/ # Cleaned & filtered datasets
│
├─ notebooks/
│ ├─ 01_data_exploration.ipynb
│ ├─ 02_network_analysis.ipynb
│ ├─ 03_feature_engineering.ipynb
│ ├─ 04_modeling.ipynb
│ └─ 05_reporting.ipynb
│
├─ src/
│ ├─ data_collection.py # Query OpenAlex & filter by UofM
│ ├─ preprocessing.py # Data cleaning & normalization
│ ├─ network_author.py # Build author co-authorship network & compute features
│ ├─ network_department.py # Build department network & compute features
│ ├─ network_topic.py # Build topic co-occurrence network
│ ├─ features.py # Combine all features into ML-ready dataset
│ ├─ models.py # Train & evaluate link prediction models
│ └─ utils.py # Helper functions & visualizations
│
├─ outputs/
│ ├─ figures/
│ ├─ models/
│ └─ reports/
│
├─ requirements.txt
└─ README.md
```

## Dependencies
Python 3.9+
pyalex
pandas, numpy
networkx or igraph
scikit-learn
matplotlib, seaborn
jupyter / jupyterlab

### Install dependencies via:

```
pip install -r requirements.txt
```

## Pipeline Overview

### Data Collection

Use pyalex to query OpenAlex.

Filter publications to UofM-affiliated authors.

### Data Cleaning

Normalize author names, extract relevant metadata (topics, departments, citations).

### Network Construction

Author Network: Nodes = authors, edges = co-authorships, edge weight = # of papers together.

Department Network: Nodes = UMich departments, edges = co-appearances on papers.

Topic Network: Nodes = research topics, edges = co-occurrences on papers.

### Feature Engineering

Network features: degree centrality, betweenness centrality, eigenvector/PageRank.

Pairwise features for ML: common neighbors, Jaccard similarity, department overlaps, topic similarity.

### Modeling

Train link prediction models (logistic regression, random forest, or graph embeddings).

Evaluate with metrics like AUC, precision@k, recall.

Visualization & Reporting

Network graphs, heatmaps, and bar charts to explore collaboration patterns.

## Usage

### Clone the repository:
```
git clone https://github.com/yourusername/UofM-collaboration-prediction.git
cd UofM-collaboration-prediction
```

### Install dependencies:

pip install -r requirements.txt

### Run notebooks or scripts in order:

01_data_exploration.ipynb

02_network_analysis.ipynb

03_feature_engineering.ipynb

04_modeling.ipynb

05_reporting.ipynb

### Notes / Limitations

The predicted collaborations are based on structural patterns in publication data, not personal or institutional factors.

All networks and features are built using publications up to the prediction year to prevent data leakage.

Future collaborations are probabilistic; performance depends on feature quality and historical patterns.

## Contributors

Kolbe Sussman – network analysis & ML modeling

Sruthi Rayasam – vizualization & reporting

Qunkun Ma – feature engineering & preprocessing
