"""
src package — Adey Innovations Fraud Detection
===============================================
Reusable modules for Task 1 (EDA & Preprocessing), Task 2 (Modelling),
and Task 3 (SHAP Explainability).
"""
from .data_loader         import (load_fraud_data, load_ip_country,
                                   load_creditcard, clean_fraud_data,
                                   clean_creditcard, merge_ip_country)
from .feature_engineering import (add_time_features, add_velocity_features,
                                   engineer_fraud_features)
from .preprocessing       import (encode_and_scale_fraud, scale_creditcard,
                                   stratified_split, apply_smote,
                                   apply_combined_resampling)
from .eda_utils           import (plot_class_imbalance, plot_numeric_distributions,
                                   plot_categorical_fraud_rate,
                                   plot_correlation_heatmap,
                                   plot_country_fraud, summarise_imbalance)
