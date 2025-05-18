"""
Module for SHAP (SHapley Additive exPlanations) analysis of the COVID-19 prediction model.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def run_shap_analysis(model_name, model, X_train, X_test, preprocessor, categorical_features, numerical_features):
    """
    Perform SHAP analysis on the trained model.
    
    Args:
        model_name (str): Name of the model.
        model: Trained model instance (Pipeline).
        X_train, X_test: Training and testing data.
        preprocessor: ColumnTransformer for preprocessing the data.
        categorical_features: List of categorical feature names.
        numerical_features: List of numerical feature names.
    """
    print(f"\n--- SHAP Analysis for {model_name} ---")

    try:
        # Try to import shap
        import shap
    except ImportError:
        print("SHAP package is not installed. Install it using: pip install shap")
        print("Skipping SHAP analysis...")
        return

    try:
        # Sample data for analysis (to avoid memory issues with SHAP)
        X_train_sample = X_train.sample(min(1000, len(X_train)), random_state=42)
        X_test_sample = X_test.sample(min(500, len(X_test)), random_state=42)
        
        # Preprocess a sample of data
        X_train_processed = model.named_steps['preprocessor'].transform(X_train_sample)
        X_test_processed = model.named_steps['preprocessor'].transform(X_test_sample)

        # Get feature names after preprocessing
        preprocessor.fit(X_train)
        cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        feature_names = list(numerical_features) + list(cat_features)

        # Ensure processed data dimensions match expected feature names
        if X_test_processed.shape[1] != len(feature_names):
            print(f"Warning: Feature count mismatch. Processed data has {X_test_processed.shape[1]} features, but feature names list has {len(feature_names)}.")
            # Adjust feature names to match the processed data size
            if X_test_processed.shape[1] < len(feature_names):
                feature_names = feature_names[:X_test_processed.shape[1]]
            else:
                # If more features than names, extend feature names with generic names
                for i in range(len(feature_names), X_test_processed.shape[1]):
                    feature_names.append(f"feature_{i}")

        # Create SHAP explainer based on model type
        if model_name in ['Random Forest', 'Gradient Boosting']:
            explainer = shap.TreeExplainer(model.named_steps['model'])
            shap_values = explainer.shap_values(X_test_processed)

            # Summary plot
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, show=False)
            plt.title(f'SHAP Summary Plot - {model_name}')
            plt.tight_layout()
            plt.show()

        elif model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet']:
            explainer = shap.LinearExplainer(model.named_steps['model'], X_train_processed)
            shap_values = explainer.shap_values(X_test_processed)

            # Summary plot
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, show=False)
            plt.title(f'SHAP Summary Plot - {model_name}')
            plt.tight_layout()
            plt.show()

        else:
            # Fallback to KernelExplainer for other model types
            background = shap.sample(X_train_processed, 50)  # Sample for efficiency
            explainer = shap.KernelExplainer(model.named_steps['model'].predict, background)
            shap_values = explainer.shap_values(shap.sample(X_test_processed, 50))

            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, shap.sample(X_test_processed, 50), feature_names=feature_names, show=False)
            plt.title(f'SHAP Summary Plot - {model_name}')
            plt.tight_layout()
            plt.show()

        # SHAP dependence plots for top features
        if model_name in ['Random Forest', 'Gradient Boosting'] or model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet']:
            # Find the top 3 features by mean absolute SHAP value
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(mean_abs_shap)[-3:]
            
            print("\nTop 3 features by SHAP value:")
            for i in top_indices:
                if i < len(feature_names):
                    print(f"  {feature_names[i]}: {mean_abs_shap[i]:.6f}")

            # Create dependence plots for top features
            for idx in top_indices:
                if idx < len(feature_names):
                    plt.figure(figsize=(10, 7))
                    shap.dependence_plot(idx, shap_values, X_test_processed, feature_names=feature_names, show=False)
                    plt.title(f'SHAP Dependence Plot - {feature_names[idx]}')
                    plt.tight_layout()
                    plt.show()

    except Exception as e:
        print(f"Error in SHAP analysis: {str(e)}")
        print("Skipping SHAP analysis and continuing with other evaluations...")

if __name__ == "__main__":
    # If running this script directly, load the data, train models and perform SHAP analysis
    from merge_data import merge_data
    from feature_engineering import engineer_features
    from model_train import train_models
    from hypertuning import tune_models
    
    df = merge_data()
    X_engineered, y, existing_demo_socio_cols = engineer_features(df)
    X_train, X_test, y_train, y_test, preprocessor, categorical_features, numerical_features, model_results = train_models(X_engineered, y)
    tuned_models, tuning_results, best_model_name, best_model = tune_models(X_train, X_test, y_train, y_test, preprocessor, model_results)
    
    run_shap_analysis(best_model_name, best_model, X_train, X_test, preprocessor, categorical_features, numerical_features)