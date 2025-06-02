"""
Hyperparameter tuning module for lasso using Optuna.
Includes a preprocessing pipeline and avoids convergence issues.
"""

import numpy as np
import optuna
from sklearn.linear_model import  Lasso
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression

from .pipeline_utils import build_preprocessing_pipeline


def tune_lasso_model(X, y, categorical_features=None, numerical_features=None, n_trials=50, cv=5):
    """
    Uses Optuna to tune lasso hyperparameters (alpha),
    wrapped in a full preprocessing and modeling pipeline.

    Parameters:
        X (DataFrame): Input features.
        y (Series or array): Target variable.
        categorical_features (list): Names of categorical columns.
        numerical_features (list): Names of numerical columns. If None, inferred automatically.
        n_trials (int): Number of Optuna optimization trials.
        cv (int): Number of folds for cross-validation.

    Returns:
        final_pipeline (Pipeline): Trained pipeline with best-found parameters.
    """

    if categorical_features is None:
        categorical_features = []

    if numerical_features is None:
        numerical_features = X.select_dtypes(include=['float64', 'int64']) \
                              .columns.difference(categorical_features).tolist()

    # Preprocessing for numeric features: impute, scale, remove low variance, select all features

    preprocessor = build_preprocessing_pipeline(numerical_features, categorical_features, use_feature_selection=True)

    # Objective function for Optuna
    def objective(trial):
        # Suggest alpha  to test
        alpha = trial.suggest_float('alpha', 1e-3, 1.0, log=True)

        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', Lasso(
                alpha=alpha,
                max_iter=2000,
                tol=1e-3,
                random_state=42
            ))
        ])

        # Try cross-validation and return average R²
        try:
            score = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1).mean()
            return score
        except Exception as e:
            print(f"Trial failed: {e}")
            return -np.inf  # Penalize failed trials

    # Run Optuna study
    study = optuna.create_study(direction='maximize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    # Show best results
    print("Best lasso Parameters:")
    print(study.best_params)
    print(f"Best cross-validated R²: {study.best_value:.4f}")

    # Train final pipeline using best parameters from study
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Lasso(
            **study.best_params,
            max_iter=2000,
            tol=1e-3,
            random_state=42
        ))
    ])
    final_pipeline.fit(X, y)

    return final_pipeline
