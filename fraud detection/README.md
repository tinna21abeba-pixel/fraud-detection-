# Fraud Detection — Adey Innovations Inc.

End-to-end fraud detection system for e-commerce and bank credit card transactions.

## Project Structure

```
fraud-detection/
├── data/
│   ├── raw/                    # Original datasets (gitignored)
│   └── processed/              # Cleaned and feature-engineered outputs
├── notebooks/
│   ├── eda-fraud-data.ipynb    # EDA for Fraud_Data.csv
│   ├── eda-creditcard.ipynb    # EDA for creditcard.csv
│   ├── feature-engineering.ipynb  # Feature engineering + preprocessing
│   ├── modeling.ipynb          # Model training & evaluation (Task 2)
│   └── shap-explainability.ipynb  # SHAP analysis (Task 3)
├── src/
│   ├── data_loader.py          # Loading, cleaning, IP-country merge
│   ├── feature_engineering.py  # Time, velocity, and derived features
│   ├── preprocessing.py        # Encoding, scaling, SMOTE
│   └── eda_utils.py            # Reusable plotting helpers
├── models/                     # Saved model artifacts (.pkl)
├── scripts/
│   └── repro_geo.py            # Standalone geolocation reproduction script
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Running Order (Task 1)

1. `notebooks/eda-fraud-data.ipynb` — EDA and geolocation enrichment
2. `notebooks/eda-creditcard.ipynb` — EDA for credit card data
3. `notebooks/feature-engineering.ipynb` — Feature engineering, scaling, SMOTE

Each notebook saves outputs to `data/processed/` for the next stage.

## Data Files Required

Place these in `data/`:
- `Fraud_Data .csv`
- `IpAddress_to_Country.csv`
- `creditcard.csv`

## Key Design Decisions (Task 1)

| Step | Choice | Reason |
|---|---|---|
| IP mapping | `merge_asof` range lookup | O(n log n), handles numeric IP ranges correctly |
| Imbalance (e-commerce) | SMOTE (strategy=0.4) | ~9.4% fraud; moderate oversampling preserves data |
| Imbalance (credit card) | SMOTE + UnderSampler | 0.17% fraud; combined avoids bloated training set |
| Scaling | `StandardScaler` | Centred features benefit Logistic Regression baseline |
| Test set | Never resampled | Maintains honest real-world evaluation |
