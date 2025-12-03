"""
Generate Predictions and Risk Scores

This script generates predictions for all buildings/ZIPs and saves them
with risk scores for visualization and analysis.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Generating Predictions and Risk Scores")
print("=" * 70)

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

# ============================================================================
# Option 2: Building-Level Predictions
# ============================================================================
print("\n" + "=" * 70)
print("OPTION 2: Building-Level Predictions")
print("=" * 70)

print("\n1. Loading data and training model...")
df = pd.read_csv(
    "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/features/merged_features_with_target.csv",
    dtype={"BBL": str}
)

# Prepare features
exclude_cols = ['BBL', 'Month', 'Target_ClassC_Next6Months']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].fillna(0)
y = df['Target_ClassC_Next6Months']

# Split chronologically
df['Month'] = pd.to_datetime(df['Month'])
split_date = df['Month'].quantile(0.8)
train_mask = df['Month'] < split_date

X_train = X[train_mask]
X_test = X[~train_mask]
y_train = y[train_mask]
y_test = y[~train_mask]

# Train best model (XGBoost)
print("   Training XGBoost model...")
model_option2 = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
model_option2.fit(X_train, y_train)

# Generate predictions for all data
print("   Generating predictions for all buildings...")
df['Risk_Score_ClassC'] = model_option2.predict_proba(X)[:, 1]
df['Predicted_ClassC'] = model_option2.predict(X)

# Save predictions
output_path = "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/predictions/building_predictions.csv"
os.makedirs("/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/predictions", exist_ok=True)

prediction_cols = ['BBL', 'Month', 'Risk_Score_ClassC', 'Predicted_ClassC', 'Target_ClassC_Next6Months']
df[prediction_cols].to_csv(output_path, index=False)

print(f"   Predictions saved to: {output_path}")
print(f"   Total buildings with predictions: {df['BBL'].nunique():,}")
print(f"   High-risk buildings (score > 0.5): {(df['Risk_Score_ClassC'] > 0.5).sum():,}")

# Summary statistics
print("\n   Risk Score Distribution:")
print(f"      Mean: {df['Risk_Score_ClassC'].mean():.4f}")
print(f"      Median: {df['Risk_Score_ClassC'].median():.4f}")
print(f"      Min: {df['Risk_Score_ClassC'].min():.4f}")
print(f"      Max: {df['Risk_Score_ClassC'].max():.4f}")

# ============================================================================
# Option 1: ZIP-Level Predictions
# ============================================================================
print("\n" + "=" * 70)
print("OPTION 1: ZIP Code-Level Predictions")
print("=" * 70)

print("\n2. Loading ZIP-level data and training model...")
df_zip = pd.read_csv(
    "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/cleaned/zip_level_monthly_data.csv"
)

# Prepare features
exclude_cols_zip = ['ZIP', 'Month', 'Year', 'High_Risk_ZIP', 'Eviction_Rate']
feature_cols_zip = [col for col in df_zip.columns if col not in exclude_cols_zip]
X_zip = df_zip[feature_cols_zip].fillna(0)
y_zip = df_zip['High_Risk_ZIP']

# Split chronologically
df_zip['Month'] = pd.to_datetime(df_zip['Month'])
split_date_zip = df_zip['Month'].quantile(0.8)
train_mask_zip = df_zip['Month'] < split_date_zip

X_train_zip = X_zip[train_mask_zip]
X_test_zip = X_zip[~train_mask_zip]
y_train_zip = y_zip[train_mask_zip]
y_test_zip = y_zip[~train_mask_zip]

# Train best model (LightGBM)
print("   Training LightGBM model...")
model_option1 = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
model_option1.fit(X_train_zip, y_train_zip)

# Generate predictions
print("   Generating predictions for all ZIP codes...")
df_zip['Risk_Score_HighEviction'] = model_option1.predict_proba(X_zip)[:, 1]
df_zip['Predicted_HighRisk'] = model_option1.predict(X_zip)

# Save predictions
output_path_zip = "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/predictions/zip_predictions.csv"
prediction_cols_zip = ['ZIP', 'Month', 'Risk_Score_HighEviction', 'Predicted_HighRisk', 'High_Risk_ZIP', 'Eviction_Count']
df_zip[prediction_cols_zip].to_csv(output_path_zip, index=False)

print(f"   Predictions saved to: {output_path_zip}")
print(f"   Total ZIP codes with predictions: {df_zip['ZIP'].nunique():,}")
print(f"   High-risk ZIPs (score > 0.5): {(df_zip['Risk_Score_HighEviction'] > 0.5).sum():,}")

# Summary statistics
print("\n   Risk Score Distribution:")
print(f"      Mean: {df_zip['Risk_Score_HighEviction'].mean():.4f}")
print(f"      Median: {df_zip['Risk_Score_HighEviction'].median():.4f}")
print(f"      Min: {df_zip['Risk_Score_HighEviction'].min():.4f}")
print(f"      Max: {df_zip['Risk_Score_HighEviction'].max():.4f}")

print("\n" + "=" * 70)
print("Predictions Generated Successfully!")
print("=" * 70)
print("\nFiles created:")
print(f"   - {output_path}")
print(f"   - {output_path_zip}")
print("\nThese files contain risk scores (0-1) for each building/ZIP,")
print("which can be used for visualization and mapping.")

