# File: src/eda/analysis.py
# Purpose: Exploratory Data Analysis (EDA) for merged housing + 311 dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
import os

# 1. Load dataset
df = pd.read_csv(
    "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/cleaned/merged_monthly_data.csv",
    dtype={"BBL": str},
    low_memory=False
)

# Rename columns for consistency
df.rename(columns={
    "Complaint Count": "311_Complaints",
    "Eviction Count": "Evictions",
    "Violation Count": "Violations",
    "permit_count": "DOB_Permits"
}, inplace=True)

# Convert Month column to datetime
df["Month"] = pd.to_datetime(df["Month"], errors="coerce")

# 2. Create output folder
eda_path = "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/eda"
os.makedirs(eda_path, exist_ok=True)

# 3. Correlation heatmap
corr = df[["311_Complaints", "Evictions", "Violations", "DOB_Permits"]].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation: 311 Complaints, Evictions, Violations, DOB Permits")
plt.tight_layout()
plt.savefig(os.path.join(eda_path, "correlation_heatmap.png"))
plt.close()

# 4. Monthly trends
monthly = (
    df.groupby("Month")[["311_Complaints", "Evictions", "Violations", "DOB_Permits"]]
    .sum()
    .reset_index()
)
plt.figure(figsize=(12, 6))
for col in ["311_Complaints", "Evictions", "Violations", "DOB_Permits"]:
    plt.plot(monthly["Month"], monthly[col], label=col)
plt.legend()
plt.title("Monthly Trends Over Time")
plt.xlabel("Month")
plt.ylabel("Counts")
plt.tight_layout()
plt.savefig(os.path.join(eda_path, "monthly_trends.png"))
plt.close()

# 5. Distribution histograms
for col in ["311_Complaints", "Evictions", "Violations", "DOB_Permits"]:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_path, f"distribution_{col.replace(' ', '_')}.png"))
    plt.close()

# 6. Pairplot
sns.pairplot(df[["311_Complaints", "Evictions", "Violations", "DOB_Permits"]])
plt.savefig(os.path.join(eda_path, "pairplot.png"))
plt.close()

# 7. Granger causality tests
monthly = monthly.dropna()
results = {}
pairs = [
    ("Evictions", "311_Complaints"),
    ("Violations", "311_Complaints"),
    ("Evictions", "Violations"),
]
for target, cause in pairs:
    test = grangercausalitytests(monthly[[target, cause]], maxlag=3, verbose=False)
    results[f"{cause} -> {target}"] = {
        f"lag {lag}": round(res[0]["ssr_ftest"][1], 4) for lag, res in test.items()
    }

granger_df = pd.DataFrame(results).T
granger_df.to_csv(os.path.join(eda_path, "granger_results.csv"))

# 8. Lagged scatter plot
monthly["Complaints_Lag1"] = monthly["311_Complaints"].shift(1)
plt.figure(figsize=(8, 6))
sns.scatterplot(x="Complaints_Lag1", y="Evictions", data=monthly)
plt.title("Lagged 311 Complaints vs Evictions")
plt.tight_layout()
plt.savefig(os.path.join(eda_path, "lagged_scatter.png"))
plt.close()

# 9. Top 10 BBLs by evictions
top_bbl = (
    df.groupby("BBL")[["Evictions"]]
    .sum()
    .sort_values("Evictions", ascending=False)
    .head(10)
    .reset_index()
)
plt.figure(figsize=(10, 5))
sns.barplot(x="BBL", y="Evictions", data=top_bbl)
plt.title("Top 10 BBLs with Highest Evictions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(eda_path, "top_bbl_evictions.png"))
plt.close()

# 10. Missing value heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.isna(), cbar=False)
plt.title("Missing Value Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(eda_path, "missing_values_heatmap.png"))
plt.close()

print("EDA completed. All output files saved to:", eda_path)