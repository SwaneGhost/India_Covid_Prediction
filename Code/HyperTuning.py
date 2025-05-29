"""
Hyperparameter tuning module for ElasticNet using Optuna.
"""

import numpy as np
import optuna
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder


def tune_elasticnet_model(X, y, categorical_features=None, numerical_features=None, n_trials=50, cv=5):
    """
    Tune ElasticNet hyperparameters using Optuna.

    Parameters:
        X (DataFrame): Feature matrix.
        y (Series or array): Target values.
        categorical_features (list): List of categorical column names.
        numerical_features (list): List of numerical column names.
        n_trials (int): Number of optimization trials.
        cv (int): Number of cross-validation folds.

    Returns:
        best_model (Pipeline): Trained pipeline with best hyperparameters.
    """
    if categorical_features is None:
        categorical_features = []

    if numerical_features is None:
        numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.difference(categorical_features).tolist()

    # Define preprocessing pipeline
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

    def objective(trial):
        alpha = trial.suggest_loguniform('alpha', 1e-4, 10)
        l1_ratio = trial.suggest_float('l1_ratio', 0.1, 1.0)

        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000))
        ])

        score = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1).mean()
        return score

    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print("Best ElasticNet Parameters:")
    print(study.best_params)
    print(f"Best cross-validated R²: {study.best_value:.4f}")

    # Train final model with best params
    best_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', ElasticNet(**study.best_params, max_iter=10000))
    ])
    best_model.fit(X, y)

    return best_model
