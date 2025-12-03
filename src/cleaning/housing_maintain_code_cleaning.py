import pandas as pd

# === 1. Load dataset ===
df = pd.read_csv("./dataset/Housing_Maintenance_Code_Violations.csv", low_memory=False)
print("Loaded housing violations with shape:", df.shape)

# === 2. Convert 'InspectionDate' to datetime ===
df['InspectionDate'] = pd.to_datetime(df['InspectionDate'], errors='coerce')

# === 3. Drop rows with missing or invalid BBL/Date ===
df = df.dropna(subset=['BBL', 'InspectionDate'])
df = df[df['BBL'] != 0]

# === 4. Filter date to 2019-01 through 2023-12 ===
df = df[
    (df['InspectionDate'] >= '2019-01-01') &
    (df['InspectionDate'] <= '2023-12-31')
]

# === 5. Convert BBL to string (safe for merging) ===
df['BBL'] = df['BBL'].astype('int64').astype(str)

# === 6. Extract Year-Month ===
df['Month'] = df['InspectionDate'].dt.to_period('M').astype(str)

# === 7. Group by BBL, Month, and Class to preserve violation classes ===
# First, get total violation count
agg_total = (
    df.groupby(['BBL', 'Month'])
    .size()
    .reset_index(name='Violation Count')
)

# Then, get counts by Class (A, B, C)
agg_by_class = (
    df.groupby(['BBL', 'Month', 'Class'])
    .size()
    .reset_index(name='Count')
    .pivot_table(index=['BBL', 'Month'], columns='Class', values='Count', fill_value=0)
    .reset_index()
)

# Rename class columns
class_cols = []
for col in ['A', 'B', 'C']:
    if col in agg_by_class.columns:
        agg_by_class.rename(columns={col: f'Class_{col}_Count'}, inplace=True)
        class_cols.append(f'Class_{col}_Count')

# Merge total and class-specific counts
agg_df = agg_total.merge(agg_by_class, on=['BBL', 'Month'], how='left')
agg_df[class_cols] = agg_df[class_cols].fillna(0).astype(int)
agg_df = agg_df.sort_values(['BBL', 'Month'])

# === 8. Output stats ===
print("\nMonthly aggregation complete.")
print("Shape:", agg_df.shape)
print("Columns:", agg_df.columns.tolist())
print("Sample rows:\n", agg_df.head())
print("\nClass distribution in aggregated data:")
if class_cols:
    print(agg_df[class_cols].sum())

# === 9. Save the output ===
agg_df.to_csv("./dataset/cleaned/housing_code_monthly_violations.csv", index=False)
agg_df.to_parquet("./dataset/cleaned/housing_code_monthly_violations.parquet", index=False)

print("\nSaved aggregated files:")
print(" - ./dataset/cleaned/housing_code_monthly_violations.csv")
print(" - ./dataset/cleaned/housing_code_monthly_violations.parquet")