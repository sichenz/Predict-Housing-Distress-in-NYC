"""
Create Target Variable for Option 2
Predicting Class C Violations in Next 6 Months

Uses an efficient vectorized approach with pandas merge operations.
For each building-month observation, creates a binary target:
- 1 if the building receives a Class C violation in the next 6 months
- 0 otherwise
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("Creating Target Variable for Option 2")
print("Predicting Class C Violations in Next 6 Months")
print("=" * 70)

# Load enhanced features dataset
print("\n1. Loading enhanced features dataset...")
df = pd.read_csv(
    "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/features/merged_features_enhanced.csv",
    dtype={"BBL": str}
)
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values(["BBL", "Month"]).reset_index(drop=True)

print(f"   Loaded {len(df):,} rows, {df.shape[1]} columns")
print(f"   Date range: {df['Month'].min()} to {df['Month'].max()}")

# Create target variable using merge-based approach
print("\n2. Creating target variable using vectorized merge approach...")

# Create month integer
df["MonthInt"] = (df["Month"].dt.year * 12) + df["Month"].dt.month

# Filter to buildings that have Class C violations
class_c_buildings = df[df["Class_C_Count"] > 0][["BBL", "MonthInt"]].copy()
class_c_buildings = class_c_buildings.rename(columns={"MonthInt": "ClassC_MonthInt"})

print(f"   Found {len(class_c_buildings):,} building-months with Class C violations")

# For each row, check if there's a Class C violation in next 6 months
print("   Creating forward-looking windows...")

# Create a cross-join approach: for each row, merge with future Class C violations
df["Target_ClassC_Next6Months"] = 0

# Process by building for efficiency
def process_building(group):
    """Process one building at a time"""
    group = group.sort_values("MonthInt").reset_index(drop=True)
    group["Target"] = 0
    
    # Get Class C months for this building
    bbl = group["BBL"].iloc[0]
    class_c_months = class_c_buildings[class_c_buildings["BBL"] == bbl]["ClassC_MonthInt"].values
    
    if len(class_c_months) == 0:
        return group[["Target"]]
    
    # For each month in group, check if any Class C in next 6 months
    for i in range(len(group)):
        current_month = group.loc[i, "MonthInt"]
        # Check if any Class C violation in next 6 months
        future_class_c = class_c_months[
            (class_c_months > current_month) & (class_c_months <= current_month + 6)
        ]
        if len(future_class_c) > 0:
            group.loc[i, "Target"] = 1
    
    return group[["Target"]]

# Apply to each building
print("   Processing buildings (much faster now)...")
unique_bbls = df["BBL"].unique()
total_bbls = len(unique_bbls)

result_list = []
chunk_size = 10000
processed = 0

for chunk_start in range(0, total_bbls, chunk_size):
    chunk_end = min(chunk_start + chunk_size, total_bbls)
    chunk_bbls = unique_bbls[chunk_start:chunk_end]
    
    if processed % 50000 == 0 and processed > 0:
        print(f"   Processed {processed:,} / {total_bbls:,} buildings...")
    
    chunk_df = df[df["BBL"].isin(chunk_bbls)].copy()
    chunk_result = chunk_df.groupby("BBL", group_keys=False).apply(process_building)
    result_list.append(chunk_result)
    processed += len(chunk_bbls)

# Combine results
print("   Combining results...")
result = pd.concat(result_list)
df["Target_ClassC_Next6Months"] = result["Target"].values

# Clean up
df = df.drop("MonthInt", axis=1)
df["Target_ClassC_Next6Months"] = df["Target_ClassC_Next6Months"].astype(int)

# Statistics
target_stats = df["Target_ClassC_Next6Months"].value_counts()
positive_rate = df["Target_ClassC_Next6Months"].mean() * 100

print(f"\n   Target variable created")
print(f"   Class distribution:")
print(f"     0 (No Class C in next 6 months): {target_stats.get(0, 0):,} ({100-positive_rate:.2f}%)")
print(f"     1 (Class C in next 6 months): {target_stats.get(1, 0):,} ({positive_rate:.2f}%)")

# Remove last 6 months from dataset
print("\n3. Removing last 6 months (no future data available)...")
max_date = df["Month"].max()
cutoff_date = max_date - pd.DateOffset(months=6)
df_train = df[df["Month"] <= cutoff_date].copy()

print(f"   Training data: {len(df_train):,} rows")
print(f"   Removed: {len(df) - len(df_train):,} rows")
print(f"   Training date range: {df_train['Month'].min()} to {df_train['Month'].max()}")

# Update target statistics for training set
train_target_stats = df_train["Target_ClassC_Next6Months"].value_counts()
train_positive_rate = df_train["Target_ClassC_Next6Months"].mean() * 100

print(f"\n   Training set target distribution:")
print(f"     0: {train_target_stats.get(0, 0):,} ({100-train_positive_rate:.2f}%)")
print(f"     1: {train_target_stats.get(1, 0):,} ({train_positive_rate:.2f}%)")

# Save dataset with target variable
output_path = "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/features/merged_features_with_target.csv"
df_train.to_csv(output_path, index=False)

print(f"\n4. Dataset with target variable saved to:")
print(f"   {output_path}")
print(f"   Shape: {df_train.shape}")

print("\n" + "=" * 70)
print("Target Variable Creation Complete!")
print("=" * 70)

