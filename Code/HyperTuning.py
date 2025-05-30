"""
Hyperparameter tuning module for ElasticNet using Optuna.
Includes a preprocessing pipeline and avoids convergence issues.
"""

import numpy as np
import optuna
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression


def tune_elasticnet_model(X, y, categorical_features=None, numerical_features=None, n_trials=50, cv=5):
    """
    Uses Optuna to tune ElasticNet hyperparameters (alpha and l1_ratio),
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
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('variance_threshold', VarianceThreshold(threshold=0.01)),
        ('select_k_best', SelectKBest(score_func=f_regression, k='all'))
    ])

    # Combine numerical and categorical preprocessing
    if categorical_features:
        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    else:
        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, numerical_features)
        ])

    # Objective function for Optuna
    def objective(trial):
        # Suggest alpha and l1_ratio to test
        alpha = trial.suggest_float('alpha', 1e-3, 1.0, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.1, 1.0)

        # Create pipeline with current trial's parameters
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
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
    print("Best ElasticNet Parameters:")
    print(study.best_params)
    print(f"Best cross-validated R²: {study.best_value:.4f}")

    # Train final pipeline using best parameters from study
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', ElasticNet(
            **study.best_params,
            max_iter=2000,
            tol=1e-3,
            random_state=42
        ))
    ])
    final_pipeline.fit(X, y)

    return final_pipeline
