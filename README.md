# Predictive Signs of Housing Distress in NYC - Project Guide

## Project Overview

This project predicts housing distress in New York City using administrative data from 2019-2023. Two prediction scenarios:
- **Option 1**: Predict high-risk ZIP codes (top 20% by eviction rate)
- **Option 2**: Predict if a building will receive a Class C violation in the next 6 months

## Prerequisites

- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, seaborn

Install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
```

For macOS, need:
```bash
brew install libomp  # Required for XGBoost
```

## Project Structure

```
FinalProject/
├── dataset/
│   ├── raw/              # Original CSV files (11GB)
│   ├── cleaned/           # Processed CSV files (3.7GB)
│   ├── features/          # Engineered features (2.0GB)
│   ├── models/            # Model results (CSV files)
│   ├── predictions/       # Prediction outputs (157MB)
│   ├── visualizations/    # All charts and maps (3.5MB)
│   └── eda/              # EDA outputs (352KB)
├── src/
│   ├── cleaning/          # Data cleaning scripts (6 files)
│   ├── eda/              # Exploratory data analysis (2 files)
│   ├── features/         # Feature engineering (1 file)
│   ├── models/           # Model training (4 files)
│   └── visualization/    # Visualization scripts (5 files)
├── docs/                 # Documentation (PDF files)
├── README.md            # This file - complete project guide
├── requirements.txt     # Python dependencies
└── .gitignore          # Git ignore rules
```

## How to Run the Project

### Step 1: Data Cleaning

Run cleaning scripts in order:

```bash
# 1. Clean 311 data
python3 src/cleaning/311_cleaning.py

# 2. Aggregate 311 by building/month
python3 src/cleaning/311_aggregate.py

# 3. Clean violations
python3 src/cleaning/housing_maintain_code_cleaning.py

# 4. Clean evictions
python3 src/cleaning/evictions_cleaning.py

# 5. Clean DOB permits
python3 src/cleaning/dob_cleaning.py

# 6. Merge all building-level data
python3 src/eda/merge_data.py

# 7. Aggregate to ZIP level (for Option 1)
python3 src/cleaning/zip_level_aggregation.py
```

**Output**: Cleaned datasets in `dataset/cleaned/`

### Step 2: Feature Engineering

```bash
# Create enhanced features (rolling, lag, delta, cumulative)
python3 src/features/feature_engineering.py
```

**Output**: `dataset/features/merged_features_enhanced.csv`

### Step 3: Create Target Variables

```bash
# Create target variable for Option 2 (Class C violation in next 6 months)
python3 src/models/create_target_variable.py
```

**Output**: `dataset/features/merged_features_with_target.csv`

Note: Option 1 target variable is created in Step 1 (zip_level_aggregation.py)

### Step 4: Train Models

```bash
# Train models for Option 1 (ZIP-level)
python3 src/models/train_option1.py

# Train models for Option 2 (Building-level)
python3 src/models/train_option2.py
```

**Output**: 
- `dataset/models/option1_results.csv`
- `dataset/models/option2_results.csv`

### Step 5: Generate Predictions

```bash
# Generate risk scores for all buildings and ZIPs
python3 src/models/generate_predictions.py
```

**Output**:
- `dataset/predictions/building_predictions.csv`
- `dataset/predictions/zip_predictions.csv`

### Step 6: Create Visualizations

```bash
# Create all visualizations
python3 src/visualization/create_visualizations.py

# Create geospatial maps
python3 src/visualization/create_maps.py

# Analyze prediction accuracy
python3 src/visualization/analyze_predictions.py
```

**Output**: All visualizations in `dataset/visualizations/`

### Step 7: View Results

```bash
# View predictions interactively
python3 src/visualization/view_predictions.py

# Open all visualizations (macOS)
python3 src/visualization/open_visualizations.py
```

## Quick Start (If Data Already Cleaned)

If you already have cleaned data, you can skip to model training:

```bash
# 1. Create target variable
python3 src/models/create_target_variable.py

# 2. Train models
python3 src/models/train_option1.py
python3 src/models/train_option2.py

# 3. Generate predictions
python3 src/models/generate_predictions.py

# 4. Create visualizations
python3 src/visualization/create_visualizations.py
python3 src/visualization/create_maps.py
```

## Expected Results

### Option 1 (ZIP-Level)
- Best Model: LightGBM
- AUC-ROC: 0.9633
- F1-Score: 0.8132
- Precision: 0.7722
- Recall: 0.8588

### Option 2 (Building-Level)
- Best Model: XGBoost
- AUC-ROC: 0.9332
- F1-Score: 0.6883
- Precision: 0.7065
- Recall: 0.6710

## Output Files

### Model Results
- `dataset/models/option1_results.csv` - ZIP-level model performance
- `dataset/models/option2_results.csv` - Building-level model performance

### Predictions
- `dataset/predictions/building_predictions.csv` - Risk scores for all buildings
- `dataset/predictions/zip_predictions.csv` - Risk scores for all ZIP codes

### Visualizations (15 files)
- Confusion matrices (2)
- Risk distributions (2)
- Risk by outcome (2)
- Risk over time (2)
- Top at-risk lists (2)
- Precision-recall curves (1)
- Risk vs evictions (1)
- Geospatial maps (3)

## Troubleshooting

**XGBoost Import Error**: Install libomp on macOS:
```bash
brew install libomp
pip uninstall xgboost
pip install xgboost
```

**Memory Issues**: The dataset is large (4.4M rows). Ensure you have sufficient RAM (8GB+ recommended).

**File Not Found Errors**: Make sure you run scripts in order, as each step depends on previous outputs.

## Notes

- All scripts use absolute paths - adjust if needed for your system
- Processing time: Full pipeline takes 1-2 hours depending on hardware
- Target variable creation is optimized for speed (uses vectorized operations)
- Models are saved in memory only - retrain to regenerate predictions


