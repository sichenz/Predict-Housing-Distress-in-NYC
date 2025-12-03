"""
ZIP Code Level Aggregation for Option 1
Predicting Eviction Risk by ZIP Code

This script aggregates all data sources by ZIP code and month,
creating features for ZIP-level eviction risk prediction.
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("ZIP Code Level Aggregation for Option 1")
print("=" * 60)

# ============================================================================
# 1. Aggregate 311 Complaints by ZIP Code
# ============================================================================
print("\n1. Aggregating 311 complaints by ZIP code...")
df_311 = pd.read_csv("./dataset/cleaned/311_cleaned.csv", 
                     usecols=['Created Date', 'Incident Zip', 'Complaint Type', 'BBL'],
                     low_memory=False)

# Convert date and filter
df_311['Created Date'] = pd.to_datetime(df_311['Created Date'], errors='coerce')
df_311 = df_311.dropna(subset=['Created Date', 'Incident Zip'])
df_311 = df_311[(df_311['Created Date'] >= '2019-01-01') & 
                (df_311['Created Date'] <= '2023-12-31')]

# Clean ZIP codes (convert to string, remove decimals)
df_311['ZIP'] = df_311['Incident Zip'].astype(str).str.split('.').str[0].str.strip()
df_311['Month'] = df_311['Created Date'].dt.to_period('M').astype(str)

# Aggregate total complaints by ZIP and month
zip_311_total = (
    df_311.groupby(['ZIP', 'Month'])
    .size()
    .reset_index(name='Complaint_Count')
)

# Aggregate key complaint types
key_types = ['HEAT/HOT WATER', 'Rodent', 'UNSANITARY CONDITION', 
             'WATER LEAK', 'MOLD', 'PLUMBING', 'ELECTRIC']
for comp_type in key_types:
    type_df = df_311[df_311['Complaint Type'] == comp_type]
    if len(type_df) > 0:
        type_agg = (
            type_df.groupby(['ZIP', 'Month'])
            .size()
            .reset_index(name=f'Complaint_{comp_type.replace(" ", "_").replace("/", "_")}')
        )
        zip_311_total = zip_311_total.merge(type_agg, on=['ZIP', 'Month'], how='left')

# Complaint diversity
complaint_diversity = (
    df_311.groupby(['ZIP', 'Month'])['Complaint Type']
    .nunique()
    .reset_index(name='Complaint_Diversity')
)
zip_311_total = zip_311_total.merge(complaint_diversity, on=['ZIP', 'Month'], how='left')

# Fill missing values
zip_311_total = zip_311_total.fillna(0)
print(f"   311 ZIP aggregation: {zip_311_total.shape}")

# ============================================================================
# 2. Aggregate Evictions by ZIP Code
# ============================================================================
print("\n2. Aggregating evictions by ZIP code...")
df_evictions = pd.read_csv('./dataset/Evictions.csv', 
                          usecols=['Executed Date', 'Eviction Postcode', 'BBL'],
                          low_memory=False)

# Convert date and filter
df_evictions['Executed Date'] = pd.to_datetime(df_evictions['Executed Date'], errors='coerce')
df_evictions = df_evictions.dropna(subset=['Executed Date', 'Eviction Postcode'])
df_evictions = df_evictions[(df_evictions['Executed Date'] >= '2019-01-01') & 
                           (df_evictions['Executed Date'] <= '2023-12-31')]

# Clean ZIP codes
df_evictions['ZIP'] = df_evictions['Eviction Postcode'].astype(str).str.split('.').str[0].str.strip()
df_evictions['Month'] = df_evictions['Executed Date'].dt.to_period('M').astype(str)

# Aggregate evictions by ZIP and month
zip_evictions = (
    df_evictions.groupby(['ZIP', 'Month'])
    .size()
    .reset_index(name='Eviction_Count')
)
print(f"   Evictions ZIP aggregation: {zip_evictions.shape}")

# ============================================================================
# 3. Aggregate Violations by ZIP Code (need to get ZIP from BBL)
# ============================================================================
print("\n3. Aggregating violations by ZIP code...")
# We need to map BBL to ZIP code from 311 data (which has both)
bbl_to_zip = df_311[['BBL', 'ZIP']].drop_duplicates().dropna()
bbl_to_zip = bbl_to_zip[bbl_to_zip['BBL'].notna()]
# Convert BBL to string for consistent merging
bbl_to_zip['BBL'] = bbl_to_zip['BBL'].astype(str)

# Load violations
df_violations = pd.read_csv('./dataset/cleaned/housing_code_monthly_violations.csv',
                           dtype={'BBL': str})
df_violations['BBL'] = df_violations['BBL'].astype(str)

# Merge with ZIP mapping
df_violations = df_violations.merge(bbl_to_zip, on='BBL', how='left')
df_violations = df_violations.dropna(subset=['ZIP'])

# Aggregate by ZIP and month
zip_violations = (
    df_violations.groupby(['ZIP', 'Month'])
    .agg({
        'Violation Count': 'sum',
        'Class_C_Count': 'sum'
    })
    .reset_index()
)
print(f"   Violations ZIP aggregation: {zip_violations.shape}")

# ============================================================================
# 4. Aggregate DOB Permits by ZIP Code
# ============================================================================
print("\n4. Aggregating DOB permits by ZIP code...")
df_dob = pd.read_csv('./dataset/cleaned/dob_permits_monthly.csv', dtype={'BBL': str})
df_dob['BBL'] = df_dob['BBL'].astype(str)

# Merge with ZIP mapping
df_dob = df_dob.merge(bbl_to_zip, on='BBL', how='left')
df_dob = df_dob.dropna(subset=['ZIP'])

# Aggregate by ZIP and month
zip_dob = (
    df_dob.groupby(['ZIP', 'Month'])
    .agg({'permit_count': 'sum'})
    .reset_index()
)
print(f"   DOB Permits ZIP aggregation: {zip_dob.shape}")

# ============================================================================
# 5. Merge All ZIP-Level Data
# ============================================================================
print("\n5. Merging all ZIP-level datasets...")
df_zip = zip_311_total.merge(zip_evictions, on=['ZIP', 'Month'], how='outer')
df_zip = df_zip.merge(zip_violations, on=['ZIP', 'Month'], how='outer')
df_zip = df_zip.merge(zip_dob, on=['ZIP', 'Month'], how='outer')

# Fill missing values with 0
df_zip = df_zip.fillna(0)

# Sort
df_zip = df_zip.sort_values(['ZIP', 'Month'])

print(f"   Final ZIP-level dataset: {df_zip.shape}")
print(f"   Unique ZIP codes: {df_zip['ZIP'].nunique()}")
print(f"   Date range: {df_zip['Month'].min()} to {df_zip['Month'].max()}")

# ============================================================================
# 6. Calculate Annual Eviction Rate (for target variable)
# ============================================================================
print("\n6. Calculating annual eviction rates...")
# Group by ZIP and year to calculate annual rates
df_zip['Year'] = pd.to_datetime(df_zip['Month']).dt.year

# Annual totals per ZIP
annual_totals = df_zip.groupby(['ZIP', 'Year']).agg({
    'Eviction_Count': 'sum',
    'Complaint_Count': 'sum'
}).reset_index()

# Calculate eviction rate (evictions per 1000 complaints, or per ZIP)
# For simplicity, use eviction count directly (can normalize later)
annual_totals['Eviction_Rate'] = annual_totals['Eviction_Count']

# Identify high-risk ZIPs (top 20% by eviction rate per year)
for year in annual_totals['Year'].unique():
    year_data = annual_totals[annual_totals['Year'] == year]
    threshold = year_data['Eviction_Rate'].quantile(0.8)
    annual_totals.loc[annual_totals['Year'] == year, 'High_Risk_ZIP'] = (
        annual_totals.loc[annual_totals['Year'] == year, 'Eviction_Rate'] >= threshold
    ).astype(int)

# Merge back to monthly data
df_zip = df_zip.merge(
    annual_totals[['ZIP', 'Year', 'High_Risk_ZIP', 'Eviction_Rate']],
    on=['ZIP', 'Year'],
    how='left'
)
df_zip['High_Risk_ZIP'] = df_zip['High_Risk_ZIP'].fillna(0).astype(int)

print(f"   High-risk ZIPs identified")
print(f"   High-risk ZIP percentage: {df_zip['High_Risk_ZIP'].mean() * 100:.1f}%")

# ============================================================================
# 7. Save ZIP-Level Dataset
# ============================================================================
output_path = "./dataset/cleaned/zip_level_monthly_data.csv"
df_zip.to_csv(output_path, index=False)
print(f"\nZIP-level dataset saved to: {output_path}")
print(f"   Shape: {df_zip.shape}")
print(f"   Columns: {df_zip.columns.tolist()}")

print("\n" + "=" * 60)
print("ZIP Code Aggregation Complete!")
print("=" * 60)

