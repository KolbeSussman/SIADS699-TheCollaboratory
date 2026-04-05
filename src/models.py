import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import os

# Paths
FEATURES_CSV = "data/processed/features.csv"
MODEL_OUTPUT = "data/processed/model_results.csv"
os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)

# Load features
df = pd.read_csv(FEATURES_CSV)
feature_cols = ['common_neighbors','jaccard','degree_diff','eigen_diff',
                'topic_overlap','dept_overlap','paper_diff','citation_diff']
X = df[feature_cols]
y = df['label']

print(f"Total samples: {len(df)}, Positives: {y.sum()}, Negatives: {len(y)-y.sum()}")

# Split by author to avoid leakage
all_authors = pd.unique(df[['author_1','author_2']].values.ravel())
train_authors, test_authors = train_test_split(all_authors, test_size=0.2, random_state=42)

def assign_split(row):
    if row['author_1'] in test_authors or row['author_2'] in test_authors:
        return 'test'
    else:
        return 'train'

df['split'] = df.apply(assign_split, axis=1)

X_train = df[df['split']=='train'][feature_cols]
y_train = df[df['split']=='train']['label']
X_test  = df[df['split']=='test'][feature_cols]
y_test  = df[df['split']=='test']['label']

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# Logistic Regression
print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train, y_train)
y_pred_lr = lr.predict_proba(X_test)[:,1]
auc_lr = roc_auc_score(y_test, y_pred_lr)
print(f"Logistic Regression AUC: {auc_lr:.4f}")

# Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict_proba(X_test)[:,1]
auc_rf = roc_auc_score(y_test, y_pred_rf)
print(f"Random Forest AUC: {auc_rf:.4f}")

# Optional: save predictions
df_test = df[df['split']=='test'].copy()
df_test['lr_pred'] = y_pred_lr
df_test['rf_pred'] = y_pred_rf
df_test.to_csv(MODEL_OUTPUT, index=False)
print(f"Saved test predictions to {MODEL_OUTPUT}")

# Detailed classification report
y_pred_rf_class = (y_pred_rf >= 0.5).astype(int)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf_class))