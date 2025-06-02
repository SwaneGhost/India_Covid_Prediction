"""
Main script for COVID-19 prediction using lasso.

This script does the following:
- Loads and prepares enhanced data
- Engineers features (NO log transformation on target)
- Trains a lasso model
- Tunes hyperparameters using Optuna
- Evaluates model performance
- Analyzes feature importance with SHAP

Updated to work directly with original target scale.
"""

import pandas as pd
import warnings
import os

from .HyperTuning import tune_lasso_model
from .model_train import train_lasso_model


# Suppress convergence warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='Objective did not converge')

from .pipeline_utils import (
    prepare_training_data,
    print_selected_features,
    save_model,
    get_feature_names_from_pipeline, evaluate_model
)

def main():

    # Step 1: Load and prepare dataset
    df = pd.read_csv('Data/Train/train_data.csv')
    # Step 3–5: Feature engineering
    print("\nStep 3–5: Preparing training features...")
    X_selected, y_raw, df_selected = prepare_training_data(df, target_col='cum_positive_cases', k='all')
    y = y_raw

    print(f"Final dataset shape: {X_selected.shape}")
    print(f"Target variable range: {y.min():.0f} to {y.max():.0f}")
    print(f"Features: {list(X_selected.columns)}")

    # Step 6: Train lasso model
    print("\nStep 6: Training lasso model...")
    try:
        model, X_train_final, X_test_final, y_train, y_test = train_lasso_model(df_selected)

        print("\nEvaluating predictions ...")
        evaluate_model(model, X_train_final, y_train, X_test_final, y_test, name="Baseline Lasso")

    except Exception as e:
        print(f"Model training failed: {e}")

    # Step 7: Hyperparameter tuning
    print("\nStep 7: Hyperparameter tuning...")
    categorical_features = ['state']
    numerical_features = X_selected.select_dtypes(include=['float64', 'int64']) \
                                    .columns.difference(categorical_features).tolist()

    print(f"Categorical features: {categorical_features}")
    print(f"Number of numerical features: {len(numerical_features)}")

    best_model = tune_lasso_model(
        X_selected, y,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        n_trials=30
    )

    # Step 8: Feature analysis
    print("\nStep 8: Selected Features Analysis:")
    print_selected_features(best_model, numerical_features)

    final_feature_names = get_feature_names_from_pipeline(best_model)
    if final_feature_names is not None:
        print(f"Total features in final model: {len(final_feature_names)}")

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    # Step 9: Save trained model
    save_model(best_model, f"Models/trained_lasso_model{timestamp}.joblib")

    # Step 10: Final Model Evaluation
    print("\nStep 10: Final Model Evaluation...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    best_model.fit(X_train, y_train)
    evaluate_model(best_model, X_train, y_train, X_test, y_test, name="Tuned Lasso")



