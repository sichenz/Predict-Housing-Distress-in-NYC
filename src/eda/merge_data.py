import pandas as pd
import os

# 1. Load cleaned datasets
print("Loading datasets...")
df_311 = pd.read_parquet("/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/cleaned/311_monthly_complaints.parquet")
df_evictions = pd.read_parquet("/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/cleaned/evictions_monthly.parquet")
df_violations = pd.read_parquet("/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/cleaned/housing_code_monthly_violations.parquet")
df_dob = pd.read_csv("/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/cleaned/dob_permits_monthly.csv")

print(f"311 shape: {df_311.shape}")
print(f"Evictions shape: {df_evictions.shape}")
print(f"Violations shape: {df_violations.shape}")
print(f"DOB shape: {df_dob.shape}")

# 2. Convert BBL and Month to string for merging
for df_name, df in [("311", df_311), ("Evictions", df_evictions), ("Violations", df_violations), ("DOB", df_dob)]:
    df["BBL"] = df["BBL"].astype(str)
    df["Month"] = df["Month"].astype(str)
    print(f"{df_name} columns: {df.columns.tolist()}")

# 3. Rename eviction and DOB count columns for consistency (if needed)
if "Eviction Count" not in df_evictions.columns:
    if "Count" in df_evictions.columns:
        df_evictions.rename(columns={"Count": "Eviction Count"}, inplace=True)

if "permit_count" not in df_dob.columns:
    if "Count" in df_dob.columns:
        df_dob.rename(columns={"Count": "permit_count"}, inplace=True)

# 4. Merge all datasets on BBL and Month
print("\nMerging datasets...")
df_merged = df_311.merge(df_evictions, on=["BBL", "Month"], how="outer")
df_merged = df_merged.merge(df_violations, on=["BBL", "Month"], how="outer")
df_merged = df_merged.merge(df_dob, on=["BBL", "Month"], how="outer")

# 5. Replace missing values with 0 for count columns only (preserve other data types)
count_cols = [col for col in df_merged.columns if 'Count' in col or 'count' in col or 'permit' in col.lower()]
df_merged[count_cols] = df_merged[count_cols].fillna(0)

# Convert count columns to int where appropriate
for col in count_cols:
    if df_merged[col].dtype == float:
        df_merged[col] = df_merged[col].astype(int)

# 6. Sort by BBL and Month
df_merged = df_merged.sort_values(["BBL", "Month"])

# 7. Save merged dataset
output_path = "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/cleaned/merged_monthly_data.csv"
df_merged.to_csv(output_path, index=False)
print(f"\nMerged dataset saved to: {output_path}")
print(f"Final shape: {df_merged.shape}")

# 8. Display summary
print("\nColumns in merged dataset:")
print(df_merged.columns.tolist())
print(f"\nTotal rows: {len(df_merged)}")
print(f"Unique BBLs: {df_merged['BBL'].nunique()}")
print(f"Date range: {df_merged['Month'].min()} to {df_merged['Month'].max()}")