import pandas as pd

# Load CSV file
df = pd.read_csv("./dataset/311_Service_Requests.csv", low_memory=False)

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())

# Keep elevant columns 
columns_to_keep = [
    'Created Date', 'Closed Date', 'Complaint Type', 'Descriptor',
    'Status', 'Incident Zip', 'Borough', 'Latitude', 'Longitude',
    'BBL', 'Community Board', 'City', 'Resolution Description'
]

df = df[columns_to_keep]
print("Columns after filtering: \n", df.columns.tolist())
print("Shape after column selection: \n", df.shape)

# Convert date columns to datetime
date_columns = ['Created Date', 'Closed Date']
for col in date_columns:
    df[col] = pd.to_datetime(df[col], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

# Remove rows with missing key fields (location or complaint info)
df.dropna(subset=['Complaint Type', 'Incident Zip', 'Latitude', 'Longitude'], inplace=True)

# Keep only NYC-based boroughs (ensure no out-of-region rows)
valid_boroughs = ['BROOKLYN', 'BRONX', 'MANHATTAN', 'QUEENS', 'STATEN ISLAND']
df = df[df['Borough'].isin(valid_boroughs)]

# Clean up inconsistent city names (optional sanity check)
df['City'] = df['City'].str.upper().fillna('')
df = df[df['City'].isin(['NEW YORK', 'BROOKLYN', 'BRONX', 'QUEENS', 'STATEN ISLAND']) | (df['City'] == '')]

# Remove duplicates (if same complaint logged twice)
df.drop_duplicates(inplace=True)

print("Final shape: \n", df.shape)

# Save cleaned dataset
# Save as Parquet (smaller and faster)
df.to_parquet("./dataset/311_cleaned.parquet", index=False)

# Also optional CSV version if needed
df.to_csv("./dataset/311_cleaned.csv", index=False)

print("Cleaning complete \n")
print("Saved cleaned files to ./dataset/311_cleaned.parquet and ./dataset/311_cleaned.csv")