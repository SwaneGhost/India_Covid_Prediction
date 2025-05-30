"""
Main script for COVID-19 prediction using ElasticNet.

This script does the following:
- Loads and prepares enhanced data
- Engineers features and transforms the target
- Trains a baseline ElasticNet model
- Tunes hyperparameters using Optuna
- Evaluates model performance
- Analyzes feature importance with SHAP
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from notebooks.shap_analysis import run_shap_analysis

# Suppress convergence warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='Objective did not converge')

from Code.pipeline_utils import (
    prepare_training_data,
    print_selected_features,
    save_model,
    get_feature_names_from_pipeline
)
from Code.enhanced_data import enhanced_data
from Code.model_train import train_refined_elasticnet_model
from Code.HyperTuning import tune_elasticnet_model


def main():
    # Step 1: Load and prepare dataset
    print("Step 1: Loading and preparing data...")
    df = enhanced_data()

    # Step 2: Optional EDA (commented out)
    # from notebooks.EDA import eda
    # eda(df)

    # Step 3–5: Feature engineering and target transformation
    print("\nStep 3–5: Preparing training features...")
    X_selected, y_raw, df_selected = prepare_training_data(df, target_col='cum_positive_cases', k='all')
    y = np.log1p(y_raw)  # Log-transform the target

    print(f"Final dataset shape: {X_selected.shape}")
    print(f"Features: {list(X_selected.columns)}")

    # Step 6: Train baseline model
    print("\nStep 6: Training baseline ElasticNet model...")
    try:
        model, X_train_final, X_test_final, y_train_log, y_test_log = train_refined_elasticnet_model(
            df_selected,
            split_type='by_state',
            custom_target=y
        )

        print("\nEvaluating predictions on original scale...")
        y_pred_log = model.predict(X_test_final)
        y_pred = np.expm1(y_pred_log)
        y_test_original = np.expm1(y_test_log)

        rmse = mean_squared_error(y_test_original, y_pred, squared=False)
        r2 = r2_score(y_test_original, y_pred)
        print(f"Test RMSE (original scale): {rmse:.2f}")
        print(f"Test R2 (original scale): {r2:.4f}")
    except Exception as e:
        print(f"Baseline model training failed: {e}")

    # Step 7: Hyperparameter tuning
    print("\nStep 7: Hyperparameter tuning...")
    categorical_features = ['state']
    numerical_features = X_selected.select_dtypes(include=['float64', 'int64']) \
                                    .columns.difference(categorical_features).tolist()

    print(f"Categorical features: {categorical_features}")
    print(f"Number of numerical features: {len(numerical_features)}")

    best_model = tune_elasticnet_model(
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

    # Step 9: Save trained model
    save_model(best_model, "trained_elasticnet_model.joblib")

    # Step 10: Final Model Evaluation...
    print("\nStep 10: Final Model Evaluation...")
    try:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

        best_model.fit(X_train, y_train)
        y_pred_log = best_model.predict(X_test)
        y_pred_original = np.expm1(y_pred_log)
        y_test_original = np.expm1(y_test)

        # Log-Transformed Metrics
        r2_log = r2_score(y_test, y_pred_log)
        rmse_log = mean_squared_error(y_test, y_pred_log, squared=False)
        mae_log = mean_absolute_error(y_test, y_pred_log)

        # Original Scale Metrics
        r2_original = r2_score(y_test_original, y_pred_original)
        rmse_original = mean_squared_error(y_test_original, y_pred_original, squared=False)
        mae_original = mean_absolute_error(y_test_original, y_pred_original)

        print("\nModel Performance Summary:")
        print("=" * 50)
        print("Log-Transformed Target Metrics:")
        print(f"  R² (log scale): {r2_log:.4f}")
        print(f"  RMSE (log scale): {rmse_log:.4f}")
        print(f"  MAE  (log scale): {mae_log:.4f}")

        print("\nOriginal Target Scale Metrics (for reference only):")
        print(f"  R² (original scale): {r2_original:.4f}")
        print(f"  RMSE (original scale): {rmse_original:,.2f}")
        print(f"  MAE  (original scale): {mae_original:,.2f}")
        print("=" * 50)

    except Exception as e:
        print(f"Model evaluation failed: {e}")

    # Step 11: SHAP analysis for interpretability
    print("\nStep 11: SHAP Analysis...")
    demo_features = [
        'population', 'GDP', 'area', 'density',
        'Hindu', 'Muslim', 'Christian', 'Sikhs', 'Buddhist', 'Others',
        'primary_health_centers', 'community_health_centers', 'sub_district_hospitals',
        'district_hospitals', 'public_health_facilities', 'public_beds',
        'urban_hospitals', 'urban_beds', 'rural_hospitals', 'rural_beds',
        'Male literacy rate', 'Female literacy rate', 'Average literacy rate',
        'Female to Male ratio', 'per capita'
    ]

    try:
        shap_values, explainer = run_shap_analysis(best_model, X_selected, demo_features, max_display=10)
        if shap_values is not None:
            print("SHAP analysis completed successfully")
        else:
            print("SHAP analysis failed")
    except Exception as e:
        print(f"SHAP analysis failed with error: {e}")

    print("\nAnalysis complete")


if __name__ == "__main__":
    main()
