import pandas as pd
import numpy as np

# === 1. Load the cleaned 311 dataset ===
df = pd.read_csv("./dataset/cleaned/311_cleaned.csv", low_memory=False)
print("Loaded cleaned dataset with shape:", df.shape)

# === 2. Convert Created Date to datetime ===
df['Created Date'] = pd.to_datetime(df['Created Date'], errors='coerce')

# === 3. Drop rows with missing Created Date or BBL ===
df = df.dropna(subset=['Created Date', 'BBL'])

# === 4. Clean BBL column to consistent string format (e.g., '1000010010') ===
df['BBL'] = df['BBL'].apply(lambda x: str(int(x)) if pd.notna(x) else None)

# === 5. Filter to include only data from 2019 to 2023 ===
df = df[(df['Created Date'] >= '2019-01-01') & (df['Created Date'] <= '2023-12-31')]

# === 6. Extract Month-Year from Created Date ===
df['Month'] = df['Created Date'].dt.to_period('M').astype(str)

# === 7. Aggregate: Total complaints per BBL per month ===
monthly_counts = (
    df.groupby(['BBL', 'Month'])
      .size()
      .reset_index(name='Complaint Count')
)

# === 8. Aggregate by complaint type (for diversity features) ===
# Focus on key complaint types mentioned in proposal
key_complaint_types = [
    'HEAT/HOT WATER', 'Rodent', 'UNSANITARY CONDITION', 
    'WATER LEAK', 'MOLD', 'PLUMBING', 'ELECTRIC',
    'Noise - Residential', 'Noise - Commercial'
]

# Calculate complaint diversity (number of unique complaint types per BBL-month)
print("Calculating complaint diversity...")
complaint_diversity = (
    df.groupby(['BBL', 'Month'])['Complaint Type']
    .nunique()
    .reset_index(name='Complaint_Diversity')
)

# Create complaint type breakdown for key types only
print("Creating complaint type breakdown for key types...")
for complaint_type in key_complaint_types:
    # Filter for this complaint type
    type_df = df[df['Complaint Type'] == complaint_type].copy()
    if len(type_df) > 0:
        type_counts = (
            type_df.groupby(['BBL', 'Month'])
            .size()
            .reset_index(name=f'Complaint_{complaint_type.replace(" ", "_").replace("/", "_").replace("-", "_")}')
        )
        monthly_counts = monthly_counts.merge(type_counts, on=['BBL', 'Month'], how='left')

# Merge diversity
monthly_counts = monthly_counts.merge(complaint_diversity, on=['BBL', 'Month'], how='left')

# Fill missing values - all complaint type columns and diversity
complaint_cols = [col for col in monthly_counts.columns if col.startswith('Complaint_')]
for col in complaint_cols + ['Complaint_Diversity']:
    monthly_counts[col] = monthly_counts[col].fillna(0).astype(int)

monthly_counts = monthly_counts.sort_values(['BBL', 'Month'])

# === 9. Print sample output ===
print("Monthly aggregation complete.\n")
print("Shape:", monthly_counts.shape)
print("Columns:", monthly_counts.columns.tolist())
print("Sample rows:\n", monthly_counts.head())

# === 10. Save to CSV and Parquet ===
monthly_counts.to_csv("./dataset/cleaned/311_monthly_complaints.csv", index=False)
monthly_counts.to_parquet("./dataset/cleaned/311_monthly_complaints.parquet", index=False)

print("\nSaved aggregated files:")
print(" - ./dataset/cleaned/311_monthly_complaints.csv")
print(" - ./dataset/cleaned/311_monthly_complaints.parquet")