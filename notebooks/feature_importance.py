"""
Module for analyzing feature importance in the COVID-19 prediction models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_feature_importance(model_name, model, X_train, X_test, y_train, y_test, preprocessor, categorical_features, numerical_features):
    """
    Analyze and visualize feature importance for the given model.
    
    Args:
        model_name (str): Name of the model.
        model: Trained model instance (Pipeline).
        X_train, X_test: Training and testing data.
        y_train, y_test: Training and testing target values.
        preprocessor: ColumnTransformer for preprocessing the data.
        categorical_features: List of categorical feature names.
        numerical_features: List of numerical feature names.
    """
    print(f"\n--- Feature Importance Analysis for {model_name} ---")

    if model_name in ['Random Forest', 'Gradient Boosting']:
        try:
            # Get feature names after preprocessing
            # First, fit the preprocessor to get the transformed feature names
            preprocessor.fit(X_train)

            # Get feature names after one-hot encoding
            cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
            feature_names = list(numerical_features) + list(cat_features)

            # Get importances
            importances = model.named_steps['model'].feature_importances_

            # Ensure we have the right number of features
            if len(importances) == len(feature_names):
                # Create DataFrame for feature importance
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False)

                # Plot feature importance
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))  # Top 20 features
                plt.title(f'Top 20 Feature Importance - {model_name}')
                plt.tight_layout()
                plt.show()

                print("Top 10 most important features:")
                print(importance_df.head(10))
            else:
                print(f"Feature names length ({len(feature_names)}) doesn't match importances length ({len(importances)})")

        except Exception as e:
            print(f"Error in feature importance analysis: {str(e)}")

    elif model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet']:
        try:
            # Get feature names after preprocessing
            preprocessor.fit(X_train)

            # Get feature names after one-hot encoding
            cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
            feature_names = list(numerical_features) + list(cat_features)

            # Get coefficients
            coefficients = model.named_steps['model'].coef_

            # Ensure we have the right number of features
            if len(coefficients) == len(feature_names):
                # Create DataFrame for coefficients
                coef_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': coefficients
                })

                # Sort by absolute value
                coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
                coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)

                # Plot coefficients
                plt.figure(figsize=(12, 8))
                top_20 = coef_df.head(20).copy()
                # Reorder by coefficient value for better visualization
                top_20 = top_20.sort_values('Coefficient')
                sns.barplot(x='Coefficient', y='Feature', data=top_20)
                plt.title(f'Top 20 Feature Coefficients - {model_name}')
                plt.axvline(x=0, color='r', linestyle='-')
                plt.tight_layout()
                plt.show()

                print("Top 10 features by coefficient magnitude:")
                print(coef_df[['Feature', 'Coefficient']].head(10))
            else:
                print(f"Feature names length ({len(feature_names)}) doesn't match coefficients length ({len(coefficients)})")

        except Exception as e:
            print(f"Error in coefficient analysis: {str(e)}")

    else:
        print(f"Feature importance analysis not implemented for {model_name}")

    # Make predictions on test data using the model
    y_pred = model.predict(X_test)

    # Plot actual vs predicted values
    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted Cumulative Positive Cases - {model_name}')
    plt.tight_layout()
    plt.show()

    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.show()

    # Histogram of residuals
    plt.figure(figsize=(12, 6))
    plt.hist(residuals, bins=50)
    plt.axvline(x=0, color='r', linestyle='-')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # If running this script directly, load the data, train models and analyze feature importance
    from merge_data import merge_data
    from feature_engineering import engineer_features
    from model_train import train_models
    from hypertuning import tune_models
    
    df = merge_data()
    X_engineered, y, existing_demo_socio_cols = engineer_features(df)
    X_train, X_test, y_train, y_test, preprocessor, categorical_features, numerical_features, model_results = train_models(X_engineered, y)
    tuned_models, tuning_results, best_model_name, best_model = tune_models(X_train, X_test, y_train, y_test, preprocessor, model_results)
    
    analyze_feature_importance(best_model_name, best_model, X_train, X_test, y_train, y_test, preprocessor, categorical_features, numerical_features)

    #dsfdfd