"""
Create Visualizations for Predictions

This script creates visualizations showing:
1. Risk score distributions
2. Top at-risk buildings/ZIPs
3. Time series of predictions
4. Feature importance plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

print("=" * 70)
print("Creating Prediction Visualizations")
print("=" * 70)

# Create output directory
output_dir = "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/visualizations"
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# Option 2: Building-Level Visualizations
# ============================================================================
print("\n1. Creating Option 2 (Building-Level) visualizations...")

# Load predictions
df_pred = pd.read_csv(
    "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/predictions/building_predictions.csv",
    dtype={"BBL": str}
)
df_pred['Month'] = pd.to_datetime(df_pred['Month'])

# 1. Risk Score Distribution
plt.figure(figsize=(10, 6))
plt.hist(df_pred['Risk_Score_ClassC'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Risk Score (Probability of Class C Violation)')
plt.ylabel('Frequency')
plt.title('Distribution of Risk Scores for Building-Level Predictions')
plt.axvline(df_pred['Risk_Score_ClassC'].mean(), color='red', linestyle='--', 
            label=f'Mean: {df_pred["Risk_Score_ClassC"].mean():.3f}')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'option2_risk_distribution.png'), dpi=300)
plt.close()

# 2. Top 20 At-Risk Buildings (by average risk score)
top_buildings = (
    df_pred.groupby('BBL')['Risk_Score_ClassC']
    .mean()
    .sort_values(ascending=False)
    .head(20)
    .reset_index()
)
top_buildings.columns = ['BBL', 'Avg_Risk_Score']

plt.figure(figsize=(12, 8))
bars = plt.barh(range(len(top_buildings)), top_buildings['Avg_Risk_Score'])
plt.yticks(range(len(top_buildings)), top_buildings['BBL'])
plt.xlabel('Average Risk Score')
plt.title('Top 20 Buildings with Highest Average Risk Scores')

# If all values are very close to 1.0, zoom in to show differences
if top_buildings['Avg_Risk_Score'].min() > 0.99:
    min_score = top_buildings['Avg_Risk_Score'].min() - 0.001
    max_score = 1.0
    plt.xlim(min_score, max_score)
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_buildings.iterrows()):
        plt.text(row['Avg_Risk_Score'], i, f'{row["Avg_Risk_Score"]:.4f}', 
                va='center', ha='left', fontsize=8)
else:
    plt.xlim(0, 1.0)

plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'option2_top_buildings.png'), dpi=300)
plt.close()

# 3. Risk Score Over Time
monthly_risk = df_pred.groupby('Month')['Risk_Score_ClassC'].mean().reset_index()
plt.figure(figsize=(12, 6))
plt.plot(monthly_risk['Month'], monthly_risk['Risk_Score_ClassC'], linewidth=2)
plt.xlabel('Month')
plt.ylabel('Average Risk Score')
plt.title('Average Building Risk Score Over Time')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'option2_risk_over_time.png'), dpi=300)
plt.close()

# 4. Prediction Accuracy by Risk Threshold
thresholds = np.arange(0.1, 1.0, 0.1)
precision_scores = []
recall_scores = []

for threshold in thresholds:
    preds = (df_pred['Risk_Score_ClassC'] >= threshold).astype(int)
    if preds.sum() > 0:
        precision = (df_pred.loc[preds == 1, 'Target_ClassC_Next6Months'].sum() / preds.sum())
        recall = (df_pred.loc[preds == 1, 'Target_ClassC_Next6Months'].sum() / df_pred['Target_ClassC_Next6Months'].sum())
        precision_scores.append(precision)
        recall_scores.append(recall)
    else:
        precision_scores.append(0)
        recall_scores.append(0)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision_scores, marker='o', label='Precision', linewidth=2)
plt.plot(thresholds, recall_scores, marker='s', label='Recall', linewidth=2)
plt.xlabel('Risk Score Threshold')
plt.ylabel('Score')
plt.title('Precision and Recall at Different Risk Thresholds')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'option2_precision_recall_curve.png'), dpi=300)
plt.close()

print("   Created 4 visualizations for Option 2")

# ============================================================================
# Option 1: ZIP-Level Visualizations
# ============================================================================
print("\n2. Creating Option 1 (ZIP-Level) visualizations...")

# Load ZIP predictions
df_zip_pred = pd.read_csv(
    "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/predictions/zip_predictions.csv"
)
df_zip_pred['Month'] = pd.to_datetime(df_zip_pred['Month'])

# 1. Risk Score Distribution
plt.figure(figsize=(10, 6))
plt.hist(df_zip_pred['Risk_Score_HighEviction'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Risk Score (Probability of High Eviction Rate)')
plt.ylabel('Frequency')
plt.title('Distribution of Risk Scores for ZIP Code-Level Predictions')
plt.axvline(df_zip_pred['Risk_Score_HighEviction'].mean(), color='red', linestyle='--',
            label=f'Mean: {df_zip_pred["Risk_Score_HighEviction"].mean():.3f}')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'option1_risk_distribution.png'), dpi=300)
plt.close()

# 2. Top 20 At-Risk ZIP Codes
top_zips = (
    df_zip_pred.groupby('ZIP')['Risk_Score_HighEviction']
    .mean()
    .sort_values(ascending=False)
    .head(20)
    .reset_index()
)
top_zips.columns = ['ZIP', 'Avg_Risk_Score']

plt.figure(figsize=(12, 8))
plt.barh(range(len(top_zips)), top_zips['Avg_Risk_Score'])
plt.yticks(range(len(top_zips)), top_zips['ZIP'].astype(str))
plt.xlabel('Average Risk Score')
plt.title('Top 20 ZIP Codes with Highest Average Risk Scores')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'option1_top_zips.png'), dpi=300)
plt.close()

# 3. Risk Score Over Time
monthly_risk_zip = df_zip_pred.groupby('Month')['Risk_Score_HighEviction'].mean().reset_index()
plt.figure(figsize=(12, 6))
plt.plot(monthly_risk_zip['Month'], monthly_risk_zip['Risk_Score_HighEviction'], linewidth=2)
plt.xlabel('Month')
plt.ylabel('Average Risk Score')
plt.title('Average ZIP Code Risk Score Over Time')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'option1_risk_over_time.png'), dpi=300)
plt.close()

# 4. Risk vs Actual Eviction Count
plt.figure(figsize=(10, 6))
plt.scatter(df_zip_pred['Risk_Score_HighEviction'], df_zip_pred['Eviction_Count'], 
           alpha=0.5, s=20)
plt.xlabel('Predicted Risk Score')
plt.ylabel('Actual Eviction Count')
plt.title('Predicted Risk Score vs Actual Eviction Count')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'option1_risk_vs_evictions.png'), dpi=300)
plt.close()

print("   Created 4 visualizations for Option 1")

# ============================================================================
# Summary Report
# ============================================================================
print("\n3. Creating summary report...")

summary = {
    'Option': ['Option 2 (Building-Level)', 'Option 1 (ZIP-Level)'],
    'Total_Predictions': [
        len(df_pred),
        len(df_zip_pred)
    ],
    'High_Risk_Count': [
        (df_pred['Risk_Score_ClassC'] > 0.5).sum(),
        (df_zip_pred['Risk_Score_HighEviction'] > 0.5).sum()
    ],
    'High_Risk_Percentage': [
        (df_pred['Risk_Score_ClassC'] > 0.5).sum() / len(df_pred) * 100,
        (df_zip_pred['Risk_Score_HighEviction'] > 0.5).sum() / len(df_zip_pred) * 100
    ],
    'Mean_Risk_Score': [
        df_pred['Risk_Score_ClassC'].mean(),
        df_zip_pred['Risk_Score_HighEviction'].mean()
    ]
}

summary_df = pd.DataFrame(summary)
summary_path = os.path.join(output_dir, 'prediction_summary.csv')
summary_df.to_csv(summary_path, index=False)

print(f"   Summary saved to: {summary_path}")
print("\n" + summary_df.to_string(index=False))

print("\n" + "=" * 70)
print("Visualization Complete!")
print("=" * 70)
print(f"\nAll visualizations saved to: {output_dir}")
print("\nGenerated files:")
print("   Option 2:")
print("     - option2_risk_distribution.png")
print("     - option2_top_buildings.png")
print("     - option2_risk_over_time.png")
print("     - option2_precision_recall_curve.png")
print("   Option 1:")
print("     - option1_risk_distribution.png")
print("     - option1_top_zips.png")
print("     - option1_risk_over_time.png")
print("     - option1_risk_vs_evictions.png")
print("   Summary:")
print("     - prediction_summary.csv")

