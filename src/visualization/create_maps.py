"""
Create Geospatial Visualizations

This script creates maps showing risk patterns across New York City
for both ZIP-level and building-level predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

print("=" * 70)
print("Creating Geospatial Visualizations")
print("=" * 70)

output_dir = "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/visualizations"
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# Option 1: ZIP Code-Level Risk Map
# ============================================================================
print("\n1. Creating ZIP code-level risk heatmap...")

# Load ZIP predictions
df_zip = pd.read_csv(
    "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/predictions/zip_predictions.csv"
)
df_zip['Month'] = pd.to_datetime(df_zip['Month'])

# Get latest month's predictions
latest_month = df_zip['Month'].max()
df_latest = df_zip[df_zip['Month'] == latest_month].copy()

# Calculate average risk by ZIP (for all time if needed)
df_zip_avg = df_zip.groupby('ZIP')['Risk_Score_HighEviction'].mean().reset_index()
df_zip_avg.columns = ['ZIP', 'Avg_Risk_Score']
df_zip_avg = df_zip_avg.sort_values('Avg_Risk_Score', ascending=False)

# Create risk categories
df_zip_avg['Risk_Category'] = pd.cut(
    df_zip_avg['Avg_Risk_Score'],
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low (0-0.3)', 'Medium (0.3-0.6)', 'High (0.6-1.0)']
)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Risk score distribution by ZIP
top_30 = df_zip_avg.head(30)
ax1.barh(range(len(top_30)), top_30['Avg_Risk_Score'], color='darkred')
ax1.set_yticks(range(len(top_30)))
ax1.set_yticklabels(top_30['ZIP'].astype(str))
ax1.set_xlabel('Average Risk Score')
ax1.set_title('Top 30 ZIP Codes by Average Risk Score')
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3)

# Right plot: Risk category distribution
risk_counts = df_zip_avg['Risk_Category'].value_counts().sort_index()
colors = ['green', 'orange', 'red']
ax2.bar(risk_counts.index.astype(str), risk_counts.values, color=colors)
ax2.set_xlabel('Risk Category')
ax2.set_ylabel('Number of ZIP Codes')
ax2.set_title('Distribution of ZIP Codes by Risk Category')
ax2.tick_params(axis='x', rotation=45)
for i, v in enumerate(risk_counts.values):
    ax2.text(i, v + 1, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'zip_risk_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: zip_risk_heatmap.png")

# Save ZIP risk summary
zip_summary = df_zip_avg.merge(
    df_latest[['ZIP', 'Eviction_Count']].groupby('ZIP')['Eviction_Count'].mean().reset_index(),
    on='ZIP',
    how='left'
)
zip_summary = zip_summary.sort_values('Avg_Risk_Score', ascending=False)
zip_summary.to_csv(
    os.path.join(output_dir, 'zip_risk_summary.csv'),
    index=False
)
print("   Saved: zip_risk_summary.csv")

# ============================================================================
# Option 2: Building-Level Risk Map (by ZIP aggregation)
# ============================================================================
print("\n2. Creating building-level risk map (aggregated by ZIP)...")

# Load building predictions
df_building = pd.read_csv(
    "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/predictions/building_predictions.csv",
    dtype={"BBL": str}
)
df_building['Month'] = pd.to_datetime(df_building['Month'])

# Get latest month
latest_month_building = df_building['Month'].max()
df_building_latest = df_building[df_building['Month'] == latest_month_building].copy()

# We need to map BBL to ZIP - use 311 data which has both
print("   Loading BBL to ZIP mapping...")
df_311 = pd.read_csv(
    "/Users/dtquynhanh/Documents/NYU/Predictive Analysis/FinalProject/dataset/cleaned/311_cleaned.csv",
    usecols=['BBL', 'Incident Zip'],
    low_memory=False
)
df_311 = df_311.dropna(subset=['BBL', 'Incident Zip'])
df_311['ZIP'] = df_311['Incident Zip'].astype(str).str.split('.').str[0].str.strip()

# Convert BBL to consistent 10-digit string format
def format_bbl(bbl):
    """Convert BBL to 10-digit string format"""
    try:
        if pd.isna(bbl):
            return None
        # Convert to int first to handle scientific notation, then to string
        bbl_int = int(float(bbl))
        return str(bbl_int).zfill(10)  # Pad to 10 digits
    except (ValueError, OverflowError):
        return None

df_311['BBL'] = df_311['BBL'].apply(format_bbl)
df_311 = df_311.dropna(subset=['BBL'])

bbl_to_zip = df_311[['BBL', 'ZIP']].drop_duplicates().dropna()
print(f"   Created BBL-ZIP mapping: {len(bbl_to_zip):,} pairs")

# Ensure BBL format matches (10-digit string) and filter out invalid BBLs
df_building_latest['BBL'] = df_building_latest['BBL'].astype(str)
# Filter out invalid BBLs (like "0", "nan", or empty)
df_building_latest = df_building_latest[
    (df_building_latest['BBL'] != '0') & 
    (df_building_latest['BBL'] != 'nan') & 
    (df_building_latest['BBL'].str.len() >= 8)  # Valid BBLs are at least 8 digits
].copy()
df_building_latest['BBL'] = df_building_latest['BBL'].str.zfill(10)

# Merge building predictions with ZIP codes
df_building_latest = df_building_latest.merge(bbl_to_zip, on='BBL', how='left')
df_building_latest = df_building_latest.dropna(subset=['ZIP'])

print(f"   Buildings with ZIP codes: {len(df_building_latest):,}")

# Aggregate building risk by ZIP
zip_building_risk = df_building_latest.groupby('ZIP').agg({
    'Risk_Score_ClassC': ['mean', 'count', 'max']
}).reset_index()
zip_building_risk.columns = ['ZIP', 'Avg_Building_Risk', 'Building_Count', 'Max_Building_Risk']
zip_building_risk = zip_building_risk.sort_values('Avg_Building_Risk', ascending=False)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Building-Level Risk Analysis by ZIP Code', fontsize=16, fontweight='bold', y=0.995)

# Top-left: Top 30 ZIPs by average building risk
ax = axes[0, 0]
top_30_buildings = zip_building_risk.head(30)
ax.barh(range(len(top_30_buildings)), top_30_buildings['Avg_Building_Risk'], color='darkred')
ax.set_yticks(range(len(top_30_buildings)))
ax.set_yticklabels(top_30_buildings['ZIP'].astype(str), fontsize=9)
ax.set_xlabel('Average Building Risk Score', fontsize=10)
ax.set_title('Top 30 ZIP Codes by Average Building Risk Score', fontsize=11, fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# Top-right: Building risk distribution
ax = axes[0, 1]
zip_building_risk['Risk_Category'] = pd.cut(
    zip_building_risk['Avg_Building_Risk'],
    bins=[0, 0.1, 0.3, 1.0],
    labels=['Low (0-0.1)', 'Medium (0.1-0.3)', 'High (0.3-1.0)']
)
risk_counts_building = zip_building_risk['Risk_Category'].value_counts().sort_index()
colors = ['green', 'orange', 'red']
bars = ax.bar(risk_counts_building.index.astype(str), risk_counts_building.values, color=colors)
ax.set_xlabel('Risk Category', fontsize=10)
ax.set_ylabel('Number of ZIP Codes', fontsize=10)
ax.set_title('Distribution of ZIP Codes by Building Risk Category', fontsize=11, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
for i, (bar, v) in enumerate(zip(bars, risk_counts_building.values)):
    ax.text(bar.get_x() + bar.get_width()/2, v + max(risk_counts_building.values)*0.02, 
            str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(0, max(risk_counts_building.values) * 1.1)

# Bottom-left: Number of buildings per ZIP
ax = axes[1, 0]
top_30_count = zip_building_risk.nlargest(30, 'Building_Count')
ax.barh(range(len(top_30_count)), top_30_count['Building_Count'], color='steelblue')
ax.set_yticks(range(len(top_30_count)))
ax.set_yticklabels(top_30_count['ZIP'].astype(str), fontsize=9)
ax.set_xlabel('Number of Buildings', fontsize=10)
ax.set_title('Top 30 ZIP Codes by Number of Buildings', fontsize=11, fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# Bottom-right: Risk vs Building Count scatter
ax = axes[1, 1]
ax.scatter(zip_building_risk['Building_Count'], zip_building_risk['Avg_Building_Risk'], 
          alpha=0.6, s=60, color='steelblue', edgecolors='black', linewidth=0.5)
ax.set_xlabel('Number of Buildings in ZIP', fontsize=10)
ax.set_ylabel('Average Building Risk Score', fontsize=10)
ax.set_title('Building Risk vs Number of Buildings per ZIP', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(os.path.join(output_dir, 'building_risk_by_zip.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: building_risk_by_zip.png")

# Save building risk summary
zip_building_risk.to_csv(
    os.path.join(output_dir, 'building_risk_by_zip_summary.csv'),
    index=False
)
print("   Saved: building_risk_by_zip_summary.csv")

# ============================================================================
# Risk Comparison: ZIP vs Building Level
# ============================================================================
print("\n3. Creating risk comparison visualization...")

# Merge ZIP-level and building-level risk
# Ensure ZIP codes are same type (string)
df_zip_avg['ZIP'] = df_zip_avg['ZIP'].astype(str)
zip_building_risk['ZIP'] = zip_building_risk['ZIP'].astype(str)

comparison = df_zip_avg[['ZIP', 'Avg_Risk_Score']].merge(
    zip_building_risk[['ZIP', 'Avg_Building_Risk']],
    on='ZIP',
    how='inner'
)
comparison.columns = ['ZIP', 'ZIP_Risk_Score', 'Building_Risk_Score']

# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Scatter plot
ax = axes[0]
ax.scatter(comparison['ZIP_Risk_Score'], comparison['Building_Risk_Score'], 
          alpha=0.6, s=50)
ax.set_xlabel('ZIP-Level Risk Score (Eviction Risk)')
ax.set_ylabel('Building-Level Risk Score (Class C Violation Risk)')
ax.set_title('ZIP-Level vs Building-Level Risk Scores')
ax.grid(True, alpha=0.3)

# Add correlation
correlation = comparison['ZIP_Risk_Score'].corr(comparison['Building_Risk_Score'])
ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
        transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Right: Top 20 ZIPs by both metrics
ax = axes[1]
comparison['Combined_Risk'] = (comparison['ZIP_Risk_Score'] + comparison['Building_Risk_Score']) / 2
top_combined = comparison.nlargest(20, 'Combined_Risk')
x = np.arange(len(top_combined))
width = 0.35
ax.barh(x - width/2, top_combined['ZIP_Risk_Score'], width, label='ZIP-Level Risk', color='steelblue')
ax.barh(x + width/2, top_combined['Building_Risk_Score'], width, label='Building-Level Risk', color='darkred')
ax.set_yticks(x)
ax.set_yticklabels(top_combined['ZIP'].astype(str))
ax.set_xlabel('Risk Score')
ax.set_title('Top 20 ZIPs by Combined Risk Score')
ax.legend()
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'risk_comparison_zip_vs_building.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: risk_comparison_zip_vs_building.png")

print("\n" + "=" * 70)
print("Geospatial Visualizations Complete!")
print("=" * 70)
print(f"\nAll maps saved to: {output_dir}")
print("\nGenerated files:")
print("   - zip_risk_heatmap.png")
print("   - zip_risk_summary.csv")
print("   - building_risk_by_zip.png")
print("   - building_risk_by_zip_summary.csv")
print("   - risk_comparison_zip_vs_building.png")

