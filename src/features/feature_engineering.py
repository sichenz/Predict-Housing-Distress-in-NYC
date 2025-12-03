import pandas as pd
import numpy as np

print("Loading merged dataset...")
# Load the data
df = pd.read_csv("/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/cleaned/merged_monthly_data.csv", dtype={"BBL": str})
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values(["BBL", "Month"])

print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

# Identify base count features (all columns with 'Count' or 'count' in name, plus permit_count)
base_features = [col for col in df.columns if ('Count' in col or 'count' in col) and col not in ['Complaint_Diversity']]
print(f"Base features for engineering: {base_features}")

# Rolling Features (3 and 6 month windows)
print("Creating rolling features...")
for window in [3, 6]:
    for feat in base_features:
        feat_name = feat.lower().replace(' ', '_')
        df[f"rolling_{window}mo_{feat_name}"] = (
            df.groupby("BBL")[feat].rolling(window=window, min_periods=1).sum().reset_index(0, drop=True)
        )

# Lag Features (1 month)
print("Creating lag features...")
for feat in base_features:
    feat_name = feat.lower().replace(' ', '_')
    lag_col = f"lag_1mo_{feat_name}"
    df[lag_col] = df.groupby("BBL")[feat].shift(1)
    # Fill first month per building with 0 (no previous month)
    df[lag_col] = df[lag_col].fillna(0)

# % Delta (Rate of Change)
print("Creating delta (rate of change) features...")
for feat in base_features:
    feat_name = feat.lower().replace(' ', '_')
    delta_col = f"delta_{feat_name}"
    df[delta_col] = df.groupby("BBL")[feat].pct_change()
    # Replace inf/-inf with NaN, then fill with 0 (no change from 0 to 0)
    df[delta_col] = df[delta_col].replace([np.inf, -np.inf], np.nan).fillna(0)

# Cumulative Sum
print("Creating cumulative features...")
for feat in base_features:
    feat_name = feat.lower().replace(' ', '_')
    df[f"total_{feat_name}"] = df.groupby("BBL")[feat].cumsum()

# Month Dummies
print("Creating month dummies...")
df["month_num"] = df["Month"].dt.month
df = pd.get_dummies(df, columns=["month_num"], prefix="month")

# Final check for any remaining missing values
print("\nChecking for remaining missing values...")
missing = df.isnull().sum()
if missing.sum() > 0:
    print("Filling missing values...")
    missing_cols = missing[missing > 0].index.tolist()
    print(f"Columns with missing values: {len(missing_cols)}")
    # Fill all missing with 0 (these are from outer merges where buildings don't have certain data)
    df = df.fillna(0)
    print("All missing values filled with 0")
else:
    print("No missing values found!")

# Save result
output_path = "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/features/merged_features_enhanced.csv"
df.to_csv(output_path, index=False)
print(f"\nEnhanced features saved to: {output_path}")
print(f"Final shape: {df.shape}")
print(f"Total columns: {len(df.columns)}")
print(f"\nSample columns: {df.columns.tolist()[:20]}...")