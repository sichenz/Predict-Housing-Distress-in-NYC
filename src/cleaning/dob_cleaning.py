import pandas as pd

# === Load raw dataset ===
df = pd.read_csv("./dataset/DOB_Permit_Issuance.csv")
print("RAW SHAPE:", df.shape)
print()

# === Parse Issuance Date ===
df["Issuance Date"] = pd.to_datetime(df["Issuance Date"], errors="coerce")

# === Filter date range to 2019–2023 ===
df = df[df["Issuance Date"].between("2019-01-01", "2023-12-31")]

# === Drop rows with missing BBL (BOROUGH + Block + Lot) ===
df = df.dropna(subset=["BOROUGH", "Block", "Lot"])

# === Clean BBL fields ===
df["BOROUGH"] = df["BOROUGH"].str.upper().str.strip()
df["Block"] = df["Block"].astype(str).str.zfill(5)
df["Lot"] = df["Lot"].astype(str).str.zfill(4)

# === Create standard 10-digit BBL ===
df["BBL"] = df["BOROUGH"].map({
    "MANHATTAN": "1", "BRONX": "2", "BROOKLYN": "3", "QUEENS": "4", "STATEN ISLAND": "5"
}) + df["Block"] + df["Lot"]

# === Extract Month ===
df["Month"] = df["Issuance Date"].dt.to_period("M").astype(str)

# === Group by BBL and Month ===
agg_df = df.groupby(["BBL", "Month"]).size().reset_index(name="permit_count")

# === Save cleaned + aggregated ===
agg_df.to_csv("./dataset/dob_permits_monthly.csv", index=False)
print("Saved to ./dataset/dob_permits_monthly.csv")
print("Preview:")
print(agg_df.head(10))