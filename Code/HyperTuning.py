"""
Module for tuning ElasticNet hyperparameters using GridSearchCV.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.metrics import r2_score


def tune_elasticnet_model(X, y, categorical_features=None, numerical_features=None, cv=5):
    """
    Tune ElasticNet hyperparameters using GridSearchCV.

    Parameters:
        X (DataFrame): Feature matrix
        y (Series or array): Target values
        categorical_features (list): List of categorical column names
        numerical_features (list): List of numerical column names
        cv (int): Number of cross-validation folds

    Returns:
        best_model (Pipeline): Best fitted model pipeline
    """

    if categorical_features is None:
        categorical_features = []
    if numerical_features is None:
        numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.difference(categorical_features).tolist()

    # === Preprocessing pipeline ===
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ]), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ] if categorical_features else [
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ]), numerical_features)
        ]
    )

    # === Pipeline with ElasticNet ===
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', ElasticNet(max_iter=10000))
    ])

    # === Hyperparameter grid ===
    param_grid = {
        'regressor__alpha': np.logspace(-4, 1, 10),
        'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    }

    # === Grid search ===
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='r2',
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X, y)

    print("✅ Best ElasticNet Parameters:")
    print(grid_search.best_params_)

    print(f"✅ Best CV R²: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_