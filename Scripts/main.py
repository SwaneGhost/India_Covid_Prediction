"""
Main script to run the COVID-19 prediction model for India.
This script coordinates the flow between different modules:
1. Data merging and cleaning
2. Exploratory Data Analysis
3. Feature Engineering
4. Model Training
5. Hyperparameter Tuning
6. Model Evaluation and Feature Importance
7. Summary and Results

Usage:
    python main.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Ensure the Data/Processed directory exists
os.makedirs(os.path.join("Data", "Processed"), exist_ok=True)

# Step 1: Merge and prepare data
print("Step 1: Merging and preparing data...")
# from merge_data import merge_data
# df = merge_data()
# print(f"Dataset loaded with shape: {df.shape}")
df = pd.read_csv('../Data/Processed/merged_data.csv')

# Step 2: Perform Exploratory Data Analysis
print("\nStep 2: Performing Exploratory Data Analysis...")
from notebooks.EDA import perform_eda
perform_eda(df)

# Step 3: Engineer features
print("\nStep 3: Engineering features...")
from Code.feature_engineering import engineer_features
X_engineered, y, demographics_socioeconomic_cols = engineer_features(df)

# Step 4: Train models
print("\nStep 4: Training models...")
from Code.model_train import train_models
X_train, X_test, y_train, y_test, preprocessor, categorical_features, numerical_features, model_results = train_models(X_engineered, y)
from Code.enhance_lasso import enhance_lasso

# Step 5: Enhance Lasso Regression
print("\nStep 5: Enhancing Lasso Regression...")
best_model, metrics = enhance_lasso(X_train, X_test, y_train, y_test, categorical_features, numerical_features)
best_model_name = "Enhanced Lasso"
tuning_results = {
    best_model_name: {
        'best_params': metrics['model'].get_params(),
        'rmse': metrics['rmse'],
        'r2': metrics['r2']
    }
}

# Step 6: Analyze feature importance
print("\nStep 6: Analyzing feature importance...")
from notebooks.feature_importance import analyze_feature_importance
analyze_feature_importance(best_model_name, best_model, 
                         X_train, X_test, y_train, y_test,
                         preprocessor, categorical_features, numerical_features)

# Step 7: Run SHAP analysis (if available)
try:
    print("\nStep 7: Performing SHAP analysis...")
    from shap_analysis import run_shap_analysis
    run_shap_analysis(best_model_name, best_model, X_train, X_test, 
                   y_train, y_test, preprocessor, categorical_features, numerical_features)
except ImportError:
    print("SHAP analysis module not available or has missing dependencies.")

# Step 8: Generate summary
print("\nStep 8: Generating summary...")
from results.Summary import generate_summary
generate_summary(
    best_model_name, best_model,
    model_results={},  # or pass your earlier results if needed
    tuning_results=tuning_results,
    X_train=X_train,
    preprocessor=None,  # or the one used, if needed
    categorical_features=categorical_features,
    numerical_features=numerical_features
)


print("\nAnalysis complete! Check the results directory for outputs.")