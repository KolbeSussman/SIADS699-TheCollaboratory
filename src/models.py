import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.ensemble import RandomForestClassifier

#Read in data and set feature and y cols
print("Reading in Data")
df = pd.read_csv("data/processed/features_temporal.csv")

print("Setting Feature Columns")
feature_cols = [c for c in df.columns if c not in ['author_1','author_2','label']]

X = df[feature_cols]
y = df['label']

# Simple split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Running Random Forest Model")
#Random Forest Model
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

print("Random Forrest metrics")
y_pred_prob = rf.predict_proba(X_test)[:,1]
y_pred = (y_pred_prob >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"AUC: {auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

#Logistic Regression Model
print("Running Logistic Regression Model")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Logit metrics")
y_pred_prob = model.predict_proba(X_test)[:,1]
y_pred = (y_pred_prob >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"AUC: {auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# Save dataset with predictions
print("Saving dataset with predictions")
df_out = df.copy()

# Add columns
df_out['y'] = df['label']
df_out['pred_y'] = None
df_out['pred_class'] = None
df_out['split'] = 'train'   # default everything to train

# Assign test rows
df_out.loc[X_test.index, 'split'] = 'test'
df_out.loc[X_test.index, 'pred_y'] = y_pred_prob
df_out.loc[X_test.index, 'pred_class'] = (y_pred_prob >= 0.5).astype(int)

# Save
OUTPUT_PRED = "data/processed/model_predictions.csv"
df_out.to_csv(OUTPUT_PRED, index=False)

print(f"Saved predictions to {OUTPUT_PRED}")