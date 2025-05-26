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

import pandas as pd
import numpy as np

# === Step 1: Merge and prepare data ===
print("Step 1: Merging and preparing data...")
from Code.enhanced_data import enhanced_data
df = enhanced_data()

# === Step 2: (Optional EDA) ===
# print("\nStep 2: Performing Exploratory Data Analysis...")
# from notebooks.EDA import perform_eda
# perform_eda(df)

# === Step 3: Engineer features ===
print("\nStep 3: Engineering features...")
from Code.feature_engineering import improved_feature_engineering
df = improved_feature_engineering(df)

# === Step 4: Feature Selection ===
print("\nStep 4: Selecting top features...")
from Code.feature_selection import select_top_k_features

# Prepare data for selection
X_full = df.drop(columns=['cum_positive_cases', 'date', 'dates'])
y = df['cum_positive_cases']

# Optionally: use log1p target if your model uses it
# y = np.log1p(y)

X_selected, selected_features, scores_df = select_top_k_features(X_full, y, k=20, return_scores=True)
print("📌 Top selected features:")
print(selected_features)

# Add back selected features + categorical (e.g., 'state') for training
X_selected['state'] = df['state']  # If using 'state' in training
df_selected = X_selected.copy()
df_selected['cum_positive_cases'] = y
df_selected['date'] = df['date']
df_selected['dates'] = df['dates']

# === Step 5: Train Models ===
print("\nStep 5: Training models...")
from Code.model_train import train_improved_elasticnet_model
model = train_improved_elasticnet_model(df_selected, split_type='by_state')

# # === Step 6: Hyperparameter Tuning ===
# print("\nStep 6: Hyperparameter tuning...")
# from Code.HyperTuning import tune_elasticnet_model
#
# categorical_features = ['state']
# numerical_features = X_selected.select_dtypes(include=['float64', 'int64']).columns.difference(categorical_features).tolist()
#
# # Tune on same selected feature set
# best_model = tune_elasticnet_model(X_selected, y, categorical_features, numerical_features)



# Step 6: Analyze feature importance
print("\nStep 6: Analyzing feature importance...")
from notebooks.feature_importance import run_feature_importance

importance_results = run_feature_importance(model, df_selected)


# === Step 7: SHAP Analysis ===
print("\nStep 7: Running SHAP feature importance analysis...")
from notebooks.shap_analysis import run_shap_analysis

# Use same training data used in model training (not split)
shap_values, explainer = run_shap_analysis(
    model=model,
    X_train=X_selected,  # or the DataFrame with raw features used to train
    feature_names=X_selected.columns.tolist(),  # optional
    max_display=9
)


# # Step 8: Generate summary
# print("\nStep 8: Generating summary...")
# from results.Summary import generate_summary
# generate_summary(
#     best_model_name, best_model,
#     model_results={},  # or pass your earlier results if needed
#     tuning_results=tuning_results,
#     X_train=X_train,
#     preprocessor=None,  # or the one used, if needed
#     categorical_features=categorical_features,
#     numerical_features=numerical_features
# )


print("\nAnalysis complete! Check the results directory for outputs.")