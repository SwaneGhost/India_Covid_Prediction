"""
Module for training machine learning models on the COVID-19 dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_models(X_engineered, y):
    """
    Train machine learning models on the engineered features.
    
    Args:
        X_engineered (pd.DataFrame): DataFrame with engineered features.
        y (pd.Series): Target variable (cumulative positive cases).
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor, categorical_features, numerical_features, model_results)
    """
    # Train-test split with engineered features
    X_train, X_test, y_train, y_test = train_test_split(X_engineered, y, test_size=0.2, random_state=42)
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    # Check if 'socioeconomic_cluster' column was created
    if 'socioeconomic_cluster' in X_engineered.columns:
        categorical_features = ['state', 'socioeconomic_cluster']
    else:
        categorical_features = ['state']
        print("Note: 'socioeconomic_cluster' not found, using only 'state' as categorical feature")

    # Split categorical and numerical features (make sure they exist in X_engineered)
    categorical_features = [col for col in categorical_features if col in X_engineered.columns]
    numerical_features = [col for col in X_engineered.columns if col not in categorical_features]

    # Create preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', KNNImputer(n_neighbors=5)),  # Change to KNN imputer for better handling of relationships
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )

    # Define models to compare
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'ElasticNet': ElasticNet(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    # Compare models with engineered features
    print("\n--- Model Comparison Using Engineered Demographic & Socioeconomic Features ---")
    model_results = compare_models(models, X_train, X_test, y_train, y_test, preprocessor)

    # Visualize model comparison
    model_names = list(model_results.keys())
    rmse_values = [model_results[model]['rmse'] for model in model_names]
    r2_values = [model_results[model]['r2'] for model in model_names]

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    bars = plt.bar(model_names, rmse_values, color='skyblue')
    plt.title('RMSE by Model with Engineered Features (Lower is Better)')
    plt.xticks(rotation=45)
    plt.ylabel('RMSE')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{bar.get_height():.2f}', ha='center')

    plt.subplot(1, 2, 2)
    bars = plt.bar(model_names, r2_values, color='lightgreen')
    plt.title('R² by Model with Engineered Features (Higher is Better)')
    plt.xticks(rotation=45)
    plt.ylabel('R²')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.4f}', ha='center')

    plt.tight_layout()
    plt.show()

    # Identify the best performing model based on R²
    best_model_name = max(model_results, key=lambda x: model_results[x]['r2'])
    print(f"\nBest performing model: {best_model_name} with R² = {model_results[best_model_name]['r2']:.4f}")

    return X_train, X_test, y_train, y_test, preprocessor, categorical_features, numerical_features, model_results

def compare_models(models, X_train, X_test, y_train, y_test, preprocessor):
    """
    Compare different machine learning models.
    
    Args:
        models (dict): Dictionary of machine learning models to compare.
        X_train, X_test, y_train, y_test: Training and testing data.
        preprocessor: ColumnTransformer for preprocessing the data.
        
    Returns:
        dict: Dictionary with model results.
    """
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Create pipeline with preprocessor and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Fit the model
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_test)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Store results
        results[name] = {
            'model': pipeline,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

        print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

    return results

if __name__ == "__main__":
    # If running this script directly, load the data, perform feature engineering and train models
    from merge_data import merge_data
    from feature_engineering import engineer_features
    
    df = merge_data()
    X_engineered, y, existing_demo_socio_cols = engineer_features(df)
    X_train, X_test, y_train, y_test, preprocessor, categorical_features, numerical_features, model_results = train_models(X_engineered, y)