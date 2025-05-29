"""
Main script for COVID-19 prediction using ElasticNet.
"""

import pandas as pd
import numpy as np
from Code.pipeline_utils import prepare_training_data, print_selected_features, save_model
from notebooks.EDA import eda
from Code.enhanced_data import enhanced_data
from Code.model_train import train_improved_elasticnet_model
from notebooks.shap_analysis import run_shap_analysis

# Step 1: Data Preparation
print("Step 1: Loading and preparing data...")
df = enhanced_data()

# Step 2: Exploratory Data Analysis
print("\nStep 2: Exploratory Data Analysis...")
# eda(df)

# Step 3–5: Feature Engineering, Correlation Filtering, Feature Selection
print("\nStep 3–5: Preparing training features...")
X_selected, y_raw, df_selected = prepare_training_data(df, target_col='cum_positive_cases', k=20)
y = np.log1p(y_raw)  # Log-transform the target



# Step 6: Model Training
print("\nStep 6: Training ElasticNet model...")
model, X_train_final, X_test_final, y_train_log, y_test_log = train_improved_elasticnet_model(
    df_selected,
    split_type='by_state',
    custom_target=y  # This is the log-transformed target
)

from sklearn.metrics import mean_squared_error, r2_score

print("\nEvaluating predictions on original scale...")

# Predict on test set (log scale)
y_pred_log = model.predict(X_test_final)

# Inverse transform to original scale
y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test_log)

# Evaluate
rmse = mean_squared_error(y_test_original, y_pred, squared=False)
r2 = r2_score(y_test_original, y_pred)

print(f"Test RMSE (original scale): {rmse:.2f}")
print(f"Test R² (original scale): {r2:.4f}")


# === Step 7: Hyperparameter Tuning ===
print("\nStep 7: Hyperparameter tuning...")
from Code.HyperTuning import tune_elasticnet_model

categorical_features = ['state']
numerical_features = X_selected.select_dtypes(include=['float64', 'int64']).columns.difference(categorical_features).tolist()

# Run Optuna-based tuning
best_model = tune_elasticnet_model(X_selected, y, categorical_features, numerical_features, n_trials=50)



# Step 7: Print final feature names
print_selected_features(model)

# Step 8: Save trained model
save_model(model, "trained_elasticnet_model.joblib")

# Step 9: SHAP Analysis
demo_features = [
    'population', 'GDP', 'area', 'density',
    'Hindu', 'Muslim', 'Christian', 'Sikhs', 'Buddhist', 'Others',
    'primary_health_centers', 'community_health_centers', 'sub_district_hospitals',
    'district_hospitals', 'public_health_facilities', 'public_beds',
    'urban_hospitals', 'urban_beds', 'rural_hospitals', 'rural_beds',
    'Male literacy rate %', 'Female literacy rate %', 'Average literacy rate %',
    'Female to Male ratio', 'per capita in'
]
run_shap_analysis(model, X_train_final, demo_features)

print("\nAnalysis complete.")
