import pandas as pd
import os

# === 1. Load Dataset ===
df = pd.read_csv('./dataset/Evictions.csv', low_memory=False)
print(f"Raw dataset shape: {df.shape}")
print("Columns:", df.columns.tolist())

# === 2. Convert 'Executed Date' to datetime ===
df['Executed Date'] = pd.to_datetime(df['Executed Date'], errors='coerce')

# === 3. Inspect Date Range & Raw Monthly Volume ===
df['Month'] = df['Executed Date'].dt.to_period('M').astype(str)
date_range = (df['Executed Date'].min(), df['Executed Date'].max())
print("\nDate Range of Executed Date (before filtering):")
print(date_range)

print("\nTop 30 month counts (raw):")
print(df['Month'].value_counts().sort_index().head(30))

# === 4. Filter by Time Range (2019–2023 only) ===
df = df[(df['Executed Date'] >= '2019-01-01') & (df['Executed Date'] <= '2023-12-31')]
print(f"\nAfter date filtering: {df.shape}")

# === 5. Clean BBL: Drop missing, invalid, or zero ===
missing_bbl = df['BBL'].isna().sum()
zero_bbl_str = (df['BBL'] == '0').sum()
df = df[df['BBL'].notna()]
df['BBL'] = pd.to_numeric(df['BBL'], errors='coerce')
df = df[df['BBL'].notna()]
df = df[df['BBL'] != 0]
df['BBL'] = df['BBL'].astype('int64')
print(f"Missing BBLs: {missing_bbl}")
print(f"Zero BBLs (string or numeric): {zero_bbl_str}")
print(f"After cleaning invalid BBLs: {df.shape}")

# === 6. Re-create Month Column ===
df['Month'] = df['Executed Date'].dt.to_period('M').astype(str)

# === 7. Aggregate Evictions per BBL per Month ===
agg = (
    df.groupby(['BBL', 'Month'])
      .size()
      .reset_index(name='Eviction Count')
      .sort_values(['BBL', 'Month'])
)
print("\nMonthly aggregation complete.")
print("Shape:", agg.shape)
print("Sample rows:\n", agg.head(10))

# === 8. Fill in missing BBL–Month pairs ===
# Generate all BBLs × Months (2019-01 to 2023-12)
all_months = pd.date_range(start='2019-01-01', end='2023-12-31', freq='MS').to_period('M').astype(str)
all_bb_ls = agg['BBL'].unique()
full_index = pd.MultiIndex.from_product([all_bb_ls, all_months], names=['BBL', 'Month'])

# Reindex and fill missing eviction counts with 0
agg_full = agg.set_index(['BBL', 'Month']).reindex(full_index, fill_value=0).reset_index()

print("\nAfter filling missing months per BBL:", agg_full.shape)
print("Sample rows after filling:\n", agg_full.head(10))

# === 9. Save Outputs ===
os.makedirs("./dataset", exist_ok=True)
agg_full.to_csv('./dataset/evictions_monthly.csv', index=False)
agg_full.to_parquet('./dataset/evictions_monthly.parquet', index=False)

print("\nSaved aggregated files:")
print(" - ./dataset/evictions_monthly.csv")
print(" - ./dataset/evictions_monthly.parquet")