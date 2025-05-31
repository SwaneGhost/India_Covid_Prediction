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
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from Code.HyperTuning import tune_lasso_model
from Code.model_train import train_lasso_model
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


def main():
    # Step 1: Load and prepare dataset
    print("Step 1: Loading and preparing data...")
    df = enhanced_data()

    # Step 2: Optional EDA (commented out)
    # from notebooks.EDA import eda
    # eda(df)

    # Step 3–5: Feature engineering (NO log transformation)
    print("\nStep 3–5: Preparing training features...")
    X_selected, y_raw, df_selected = prepare_training_data(df, target_col='cum_positive_cases', k='all')
    y = y_raw  # Use original scale directly - NO log transformation

    print(f"Final dataset shape: {X_selected.shape}")
    print(f"Target variable range: {y.min():.0f} to {y.max():.0f}")
    print(f"Features: {list(X_selected.columns)}")

    # Step 6: Train lasso model
    print("\nStep 6: Training lasso model...")
    try:
        model, X_train_final, X_test_final, y_train, y_test = train_lasso_model(
            df_selected,
            split_type='by_state',
            custom_target=y  # Pass original scale target
        )

        print("\nEvaluating predictions on original scale...")
        y_pred = model.predict(X_test_final)  # Direct predictions in original scale

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"Test RMSE: {rmse:,.2f}")
        print(f"Test R²: {r2:.4f}")
        print(f"Test MAE: {mae:,.2f}")

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
        X_selected, y,  # Use original scale target
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
    save_model(best_model, "trained_lasso_model.joblib")

    # Step 10: Final Model Evaluation
    print("\nStep 10: Final Model Evaluation...")
    try:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)  # Direct predictions in original scale

        # Calculate metrics in original scale
        r2_score_final = r2_score(y_test, y_pred)
        rmse_final = mean_squared_error(y_test, y_pred, squared=False)
        mae_final = mean_absolute_error(y_test, y_pred)

        # Calculate training metrics to check overfitting
        y_train_pred = best_model.predict(X_train)
        r2_train = r2_score(y_train, y_train_pred)
        rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)

        print("\nFinal Model Performance Summary:")
        print("=" * 50)
        print("Training Metrics:")
        print(f"  R²: {r2_train:.4f}")
        print(f"  RMSE: {rmse_train:,.2f}")

        print("\nTest Metrics:")
        print(f"  R²: {r2_score_final:.4f}")
        print(f"  RMSE: {rmse_final:,.2f}")
        print(f"  MAE: {mae_final:,.2f}")

        print(f"\nOverfitting Check:")
        print(f"  R² difference (train - test): {r2_train - r2_score_final:.4f}")

        print(f"\nPrediction Range:")
        print(f"  Min prediction: {y_pred.min():,.0f}")
        print(f"  Max prediction: {y_pred.max():,.0f}")
        print(f"  Actual range: {y_test.min():,.0f} to {y_test.max():,.0f}")
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
            print("Note: SHAP values now represent direct impact on COVID case counts (original scale)")
        else:
            print("SHAP analysis failed")
    except Exception as e:
        print(f"SHAP analysis failed with error: {e}")

    print("\nAnalysis complete - All metrics in original COVID case count scale")


if __name__ == "__main__":
    main()