"""
Analyze Prediction Accuracy and Relationships

This script analyzes how well predictions match actual outcomes
and creates summary reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report

print("=" * 70)
print("Analyzing Prediction Accuracy and Relationships")
print("=" * 70)

output_dir = "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/visualizations"
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# Option 2: Building-Level Analysis
# ============================================================================
print("\n" + "=" * 70)
print("OPTION 2: Building-Level Prediction Analysis")
print("=" * 70)

df_building = pd.read_csv(
    "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/predictions/building_predictions.csv",
    dtype={"BBL": str}
)
df_building['Month'] = pd.to_datetime(df_building['Month'])

# Confusion Matrix
print("\n1. Confusion Matrix (Predicted vs Actual):")
cm = confusion_matrix(df_building['Target_ClassC_Next6Months'], df_building['Predicted_ClassC'])
print(f"\n   True Negatives:  {cm[0,0]:,}")
print(f"   False Positives: {cm[0,1]:,}")
print(f"   False Negatives: {cm[1,0]:,}")
print(f"   True Positives:  {cm[1,1]:,}")

# Calculate accuracy metrics
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"   F1-Score: {f1:.4f}")

# Visualize Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Violation', 'Violation'],
            yticklabels=['No Violation', 'Violation'])
plt.title('Confusion Matrix: Building-Level Predictions\n(Predicted Class C Violation vs Actual)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'option2_confusion_matrix.png'), dpi=300)
plt.close()
print("   Saved: option2_confusion_matrix.png")

# Risk Score vs Actual Outcome
print("\n2. Risk Score Distribution by Actual Outcome:")
print("\n   Buildings WITH Class C Violations:")
print(f"      Mean Risk Score: {df_building[df_building['Target_ClassC_Next6Months']==1]['Risk_Score_ClassC'].mean():.4f}")
print(f"      Median: {df_building[df_building['Target_ClassC_Next6Months']==1]['Risk_Score_ClassC'].median():.4f}")

print("\n   Buildings WITHOUT Class C Violations:")
print(f"      Mean Risk Score: {df_building[df_building['Target_ClassC_Next6Months']==0]['Risk_Score_ClassC'].mean():.4f}")
print(f"      Median: {df_building[df_building['Target_ClassC_Next6Months']==0]['Risk_Score_ClassC'].median():.4f}")

# Visualize Risk Score Distribution by Outcome
plt.figure(figsize=(12, 6))
plt.hist(df_building[df_building['Target_ClassC_Next6Months']==0]['Risk_Score_ClassC'], 
         bins=50, alpha=0.6, label='No Violation', color='green', density=True)
plt.hist(df_building[df_building['Target_ClassC_Next6Months']==1]['Risk_Score_ClassC'], 
         bins=50, alpha=0.6, label='Has Violation', color='red', density=True)
plt.xlabel('Risk Score')
plt.ylabel('Density')
plt.title('Risk Score Distribution: Predicted vs Actual Outcome')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'option2_risk_by_outcome.png'), dpi=300)
plt.close()
print("   Saved: option2_risk_by_outcome.png")

# Correlation Analysis
print("\n3. Correlation Analysis:")
correlation = df_building['Risk_Score_ClassC'].corr(df_building['Target_ClassC_Next6Months'])
print(f"   Correlation between Risk Score and Actual Outcome: {correlation:.4f}")
print(f"   Interpretation: {'Strong positive relationship' if correlation > 0.5 else 'Moderate relationship' if correlation > 0.3 else 'Weak relationship'}")

# Accuracy by Risk Threshold
print("\n4. Prediction Accuracy at Different Risk Thresholds:")
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
results = []

for threshold in thresholds:
    preds = (df_building['Risk_Score_ClassC'] >= threshold).astype(int)
    if preds.sum() > 0:
        cm_thresh = confusion_matrix(df_building['Target_ClassC_Next6Months'], preds)
        if cm_thresh.size == 4:
            tn, fp, fn, tp = cm_thresh.ravel()
            precision_thresh = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_thresh = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_thresh = 2 * (precision_thresh * recall_thresh) / (precision_thresh + recall_thresh) if (precision_thresh + recall_thresh) > 0 else 0
            results.append({
                'Threshold': threshold,
                'Precision': precision_thresh,
                'Recall': recall_thresh,
                'F1': f1_thresh,
                'TP': tp,
                'FP': fp,
                'FN': fn
            })

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# ============================================================================
# Option 1: ZIP-Level Analysis
# ============================================================================
print("\n\n" + "=" * 70)
print("OPTION 1: ZIP Code-Level Prediction Analysis")
print("=" * 70)

df_zip = pd.read_csv(
    "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/predictions/zip_predictions.csv"
)
df_zip['Month'] = pd.to_datetime(df_zip['Month'])

# Confusion Matrix
print("\n1. Confusion Matrix (Predicted vs Actual):")
cm_zip = confusion_matrix(df_zip['High_Risk_ZIP'], df_zip['Predicted_HighRisk'])
print(f"\n   True Negatives:  {cm_zip[0,0]:,}")
print(f"   False Positives: {cm_zip[0,1]:,}")
print(f"   False Negatives: {cm_zip[1,0]:,}")
print(f"   True Positives:  {cm_zip[1,1]:,}")

tn_zip, fp_zip, fn_zip, tp_zip = cm_zip.ravel()
accuracy_zip = (tp_zip + tn_zip) / (tp_zip + tn_zip + fp_zip + fn_zip)
precision_zip = tp_zip / (tp_zip + fp_zip) if (tp_zip + fp_zip) > 0 else 0
recall_zip = tp_zip / (tp_zip + fn_zip) if (tp_zip + fn_zip) > 0 else 0
f1_zip = 2 * (precision_zip * recall_zip) / (precision_zip + recall_zip) if (precision_zip + recall_zip) > 0 else 0

print(f"\n   Accuracy:  {accuracy_zip:.4f} ({accuracy_zip*100:.2f}%)")
print(f"   Precision: {precision_zip:.4f} ({precision_zip*100:.2f}%)")
print(f"   Recall:    {recall_zip:.4f} ({recall_zip*100:.2f}%)")
print(f"   F1-Score: {f1_zip:.4f}")

# Visualize Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_zip, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not High-Risk', 'High-Risk'],
            yticklabels=['Not High-Risk', 'High-Risk'])
plt.title('Confusion Matrix: ZIP Code-Level Predictions\n(Predicted High-Risk ZIP vs Actual)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'option1_confusion_matrix.png'), dpi=300)
plt.close()
print("   Saved: option1_confusion_matrix.png")

# Risk Score vs Actual Outcome
print("\n2. Risk Score Distribution by Actual Outcome:")
print("\n   ZIPs WITH High Eviction Risk:")
print(f"      Mean Risk Score: {df_zip[df_zip['High_Risk_ZIP']==1]['Risk_Score_HighEviction'].mean():.4f}")
print(f"      Median: {df_zip[df_zip['High_Risk_ZIP']==1]['Risk_Score_HighEviction'].median():.4f}")

print("\n   ZIPs WITHOUT High Eviction Risk:")
print(f"      Mean Risk Score: {df_zip[df_zip['High_Risk_ZIP']==0]['Risk_Score_HighEviction'].mean():.4f}")
print(f"      Median: {df_zip[df_zip['High_Risk_ZIP']==0]['Risk_Score_HighEviction'].median():.4f}")

# Visualize Risk Score Distribution by Outcome
plt.figure(figsize=(12, 6))
plt.hist(df_zip[df_zip['High_Risk_ZIP']==0]['Risk_Score_HighEviction'], 
         bins=30, alpha=0.6, label='Not High-Risk', color='green', density=True)
plt.hist(df_zip[df_zip['High_Risk_ZIP']==1]['Risk_Score_HighEviction'], 
         bins=30, alpha=0.6, label='High-Risk', color='red', density=True)
plt.xlabel('Risk Score')
plt.ylabel('Density')
plt.title('Risk Score Distribution: Predicted vs Actual Outcome')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'option1_risk_by_outcome.png'), dpi=300)
plt.close()
print("   Saved: option1_risk_by_outcome.png")

# Correlation Analysis
print("\n3. Correlation Analysis:")
correlation_zip = df_zip['Risk_Score_HighEviction'].corr(df_zip['High_Risk_ZIP'])
print(f"   Correlation between Risk Score and Actual Outcome: {correlation_zip:.4f}")
print(f"   Interpretation: {'Strong positive relationship' if correlation_zip > 0.5 else 'Moderate relationship' if correlation_zip > 0.3 else 'Weak relationship'}")

# Risk Score vs Eviction Count
print("\n4. Risk Score vs Actual Eviction Count:")
eviction_corr = df_zip['Risk_Score_HighEviction'].corr(df_zip['Eviction_Count'])
print(f"   Correlation: {eviction_corr:.4f}")

# ============================================================================
# Summary Report
# ============================================================================
print("\n\n" + "=" * 70)
print("SUMMARY: Are Predictions Related to Actual Outcomes?")
print("=" * 70)

print("\nOPTION 2 (Building-Level):")
print(f"   • Correlation: {correlation:.4f} - {'STRONG' if correlation > 0.5 else 'MODERATE' if correlation > 0.3 else 'WEAK'} relationship")
print(f"   • Accuracy: {accuracy*100:.2f}%")
print(f"   • High-risk buildings (score > 0.5) have {df_building[df_building['Risk_Score_ClassC']>0.5]['Target_ClassC_Next6Months'].mean()*100:.1f}% actual violation rate")
print(f"   • Low-risk buildings (score < 0.3) have {df_building[df_building['Risk_Score_ClassC']<0.3]['Target_ClassC_Next6Months'].mean()*100:.1f}% actual violation rate")

print("\nOPTION 1 (ZIP-Level):")
print(f"   • Correlation: {correlation_zip:.4f} - {'STRONG' if correlation_zip > 0.5 else 'MODERATE' if correlation_zip > 0.3 else 'WEAK'} relationship")
print(f"   • Accuracy: {accuracy_zip*100:.2f}%")
print(f"   • High-risk ZIPs (score > 0.5) have {df_zip[df_zip['Risk_Score_HighEviction']>0.5]['High_Risk_ZIP'].mean()*100:.1f}% actual high-risk rate")
print(f"   • Low-risk ZIPs (score < 0.3) have {df_zip[df_zip['Risk_Score_HighEviction']<0.3]['High_Risk_ZIP'].mean()*100:.1f}% actual high-risk rate")

print("\n" + "=" * 70)
print("Analysis Complete! New visualizations saved.")
print("=" * 70)
print("\nNew files created:")
print("   - option2_confusion_matrix.png")
print("   - option2_risk_by_outcome.png")
print("   - option1_confusion_matrix.png")
print("   - option1_risk_by_outcome.png")
print("\nThese show how well predictions match actual outcomes!")

