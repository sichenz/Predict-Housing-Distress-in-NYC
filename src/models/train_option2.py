"""
Train Models for Option 2: Building-Level Class C Violation Prediction

This script trains multiple models to predict if a building will receive
a Class C violation in the next 6 months.

Models:
- Logistic Regression (baseline)
- XGBoost
- LightGBM
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, but continue if it fails
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception as e:
    print(f"Warning: XGBoost not available ({e}). Continuing without it.")
    XGBOOST_AVAILABLE = False

print("=" * 70)
print("Training Models for Option 2: Building-Level Prediction")
print("Predicting Class C Violations in Next 6 Months")
print("=" * 70)

# Load dataset with target variable
print("\n1. Loading dataset with target variable...")
df = pd.read_csv(
    "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/features/merged_features_with_target.csv",
    dtype={"BBL": str}
)

print(f"   Loaded {len(df):,} rows, {df.shape[1]} columns")

# Prepare features and target
print("\n2. Preparing features and target...")

# Exclude non-feature columns
exclude_cols = ['BBL', 'Month', 'Target_ClassC_Next6Months']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df['Target_ClassC_Next6Months']

print(f"   Features: {len(feature_cols)}")
print(f"   Target distribution: {y.value_counts().to_dict()}")

# Handle any remaining missing values
print("\n3. Handling missing values...")
missing = X.isnull().sum().sum()
if missing > 0:
    print(f"   Found {missing} missing values, filling with 0")
    X = X.fillna(0)
else:
    print("   No missing values")

# Split data chronologically (use Month for time-based split)
print("\n4. Splitting data (chronological split)...")
df['Month'] = pd.to_datetime(df['Month'])
split_date = df['Month'].quantile(0.8)  # 80% for training, 20% for testing

train_mask = df['Month'] < split_date
X_train = X[train_mask]
X_test = X[~train_mask]
y_train = y[train_mask]
y_test = y[~train_mask]

print(f"   Training set: {len(X_train):,} rows ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Test set: {len(X_test):,} rows ({len(X_test)/len(X)*100:.1f}%)")
print(f"   Split date: {split_date.date()}")

# Train Logistic Regression (baseline)
print("\n" + "=" * 70)
print("5. Training Logistic Regression (Baseline)...")
print("=" * 70)

lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_pred_proba_lr = lr.predict_proba(X_test)[:, 1]

lr_auc = roc_auc_score(y_test, y_pred_proba_lr)
lr_f1 = f1_score(y_test, y_pred_lr)
lr_precision = precision_score(y_test, y_pred_lr)
lr_recall = recall_score(y_test, y_pred_lr)

print(f"\nLogistic Regression Results:")
print(f"   AUC-ROC: {lr_auc:.4f}")
print(f"   F1-Score: {lr_f1:.4f}")
print(f"   Precision: {lr_precision:.4f}")
print(f"   Recall: {lr_recall:.4f}")

# Train XGBoost (if available)
if XGBOOST_AVAILABLE:
    print("\n" + "=" * 70)
    print("6. Training XGBoost...")
    print("=" * 70)

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    xgb_model.fit(X_train, y_train)

    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

    xgb_auc = roc_auc_score(y_test, y_pred_proba_xgb)
    xgb_f1 = f1_score(y_test, y_pred_xgb)
    xgb_precision = precision_score(y_test, y_pred_xgb)
    xgb_recall = recall_score(y_test, y_pred_xgb)

    print(f"\nXGBoost Results:")
    print(f"   AUC-ROC: {xgb_auc:.4f}")
    print(f"   F1-Score: {xgb_f1:.4f}")
    print(f"   Precision: {xgb_precision:.4f}")
    print(f"   Recall: {xgb_recall:.4f}")
else:
    print("\n" + "=" * 70)
    print("6. Skipping XGBoost (not available)")
    print("=" * 70)
    xgb_auc = xgb_f1 = xgb_precision = xgb_recall = None
    xgb_model = None

# Train LightGBM
print("\n" + "=" * 70)
print("7. Training LightGBM...")
print("=" * 70)

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_model.fit(X_train, y_train)

y_pred_lgb = lgb_model.predict(X_test)
y_pred_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]

lgb_auc = roc_auc_score(y_test, y_pred_proba_lgb)
lgb_f1 = f1_score(y_test, y_pred_lgb)
lgb_precision = precision_score(y_test, y_pred_lgb)
lgb_recall = recall_score(y_test, y_pred_lgb)

print(f"\nLightGBM Results:")
print(f"   AUC-ROC: {lgb_auc:.4f}")
print(f"   F1-Score: {lgb_f1:.4f}")
print(f"   Precision: {lgb_precision:.4f}")
print(f"   Recall: {lgb_recall:.4f}")

# Summary
print("\n" + "=" * 70)
print("MODEL COMPARISON SUMMARY")
print("=" * 70)

models = ['Logistic Regression']
aucs = [lr_auc]
f1s = [lr_f1]
precisions = [lr_precision]
recalls = [lr_recall]

if XGBOOST_AVAILABLE:
    models.append('XGBoost')
    aucs.append(xgb_auc)
    f1s.append(xgb_f1)
    precisions.append(xgb_precision)
    recalls.append(xgb_recall)

models.append('LightGBM')
aucs.append(lgb_auc)
f1s.append(lgb_f1)
precisions.append(lgb_precision)
recalls.append(lgb_recall)

results_df = pd.DataFrame({
    'Model': models,
    'AUC-ROC': aucs,
    'F1-Score': f1s,
    'Precision': precisions,
    'Recall': recalls
})

print("\n" + results_df.to_string(index=False))

# Save results
results_path = "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/models/option2_results.csv"
import os
os.makedirs("/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/models", exist_ok=True)
results_df.to_csv(results_path, index=False)

print(f"\nResults saved to: {results_path}")

# Feature importance (for tree-based models)
if XGBOOST_AVAILABLE:
    print("\n" + "=" * 70)
    print("TOP 10 MOST IMPORTANT FEATURES (XGBoost)")
    print("=" * 70)

    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)

    print("\n" + feature_importance.to_string(index=False))

print("\n" + "=" * 70)
print("TOP 10 MOST IMPORTANT FEATURES (LightGBM)")
print("=" * 70)

feature_importance_lgb = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': lgb_model.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

print("\n" + feature_importance_lgb.to_string(index=False))

print("\n" + "=" * 70)
print("Model Training Complete!")
print("=" * 70)

