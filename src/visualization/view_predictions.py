"""
View Predictions - Interactive Script

This script shows how to view and interpret predictions.
Run this to see examples of predictions for specific buildings/ZIPs.
"""

import pandas as pd

print("=" * 70)
print("Viewing Predictions")
print("=" * 70)

# ============================================================================
# Option 2: Building-Level Predictions
# ============================================================================
print("\n" + "=" * 70)
print("OPTION 2: Building-Level Predictions")
print("=" * 70)

df_building = pd.read_csv(
    "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/predictions/building_predictions.csv",
    dtype={"BBL": str}
)
df_building['Month'] = pd.to_datetime(df_building['Month'])

print(f"\nTotal predictions: {len(df_building):,}")
print(f"Unique buildings: {df_building['BBL'].nunique():,}")

# Show highest risk buildings
print("\nTop 10 Buildings with Highest Risk Scores (Most Recent Month):")
latest_month = df_building['Month'].max()
latest_predictions = df_building[df_building['Month'] == latest_month].copy()
top_risky = latest_predictions.nlargest(10, 'Risk_Score_ClassC')[
    ['BBL', 'Month', 'Risk_Score_ClassC', 'Predicted_ClassC', 'Target_ClassC_Next6Months']
]
print(top_risky.to_string(index=False))

# Risk categories
print("\n\nRisk Categories (for latest month):")
latest_predictions['Risk_Category'] = pd.cut(
    latest_predictions['Risk_Score_ClassC'],
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low Risk (0-0.3)', 'Medium Risk (0.3-0.6)', 'High Risk (0.6-1.0)']
)
risk_summary = latest_predictions['Risk_Category'].value_counts()
print(risk_summary)

# ============================================================================
# Option 1: ZIP-Level Predictions
# ============================================================================
print("\n\n" + "=" * 70)
print("OPTION 1: ZIP Code-Level Predictions")
print("=" * 70)

df_zip = pd.read_csv(
    "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/predictions/zip_predictions.csv"
)
df_zip['Month'] = pd.to_datetime(df_zip['Month'])

print(f"\nTotal predictions: {len(df_zip):,}")
print(f"Unique ZIP codes: {df_zip['ZIP'].nunique()}")

# Show highest risk ZIPs
print("\nTop 10 ZIP Codes with Highest Risk Scores (Most Recent Month):")
latest_month_zip = df_zip['Month'].max()
latest_predictions_zip = df_zip[df_zip['Month'] == latest_month_zip].copy()
top_risky_zip = latest_predictions_zip.nlargest(10, 'Risk_Score_HighEviction')[
    ['ZIP', 'Month', 'Risk_Score_HighEviction', 'Predicted_HighRisk', 'High_Risk_ZIP', 'Eviction_Count']
]
print(top_risky_zip.to_string(index=False))

# Risk categories
print("\n\nRisk Categories (for latest month):")
latest_predictions_zip['Risk_Category'] = pd.cut(
    latest_predictions_zip['Risk_Score_HighEviction'],
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low Risk (0-0.3)', 'Medium Risk (0.3-0.6)', 'High Risk (0.6-1.0)']
)
risk_summary_zip = latest_predictions_zip['Risk_Category'].value_counts()
print(risk_summary_zip)

print("\n" + "=" * 70)
print("How to Interpret Predictions:")
print("=" * 70)
print("\nRisk Score: Probability (0-1) that the event will occur")
print("  - 0.0-0.3: Low Risk")
print("  - 0.3-0.6: Medium Risk")
print("  - 0.6-1.0: High Risk")
print("\nPredicted: Binary prediction (1 = will occur, 0 = won't occur)")
print("  - Based on threshold of 0.5")
print("\nUse these predictions to:")
print("  - Identify at-risk buildings for early intervention")
print("  - Prioritize ZIP codes for resource allocation")
print("  - Track risk trends over time")

