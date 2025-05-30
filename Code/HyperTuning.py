"""
Hyperparameter tuning module for ElasticNet using Optuna.
Fixed to handle pipeline structure correctly and avoid convergence warnings.
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
    Tune ElasticNet hyperparameters using Optuna and return a full pipeline.

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

    # Define preprocessing for numerical features
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('variance_threshold', VarianceThreshold(threshold=0.01)),
        ('select_k_best', SelectKBest(score_func=f_regression, k='all'))
    ])

    # Define preprocessing pipeline
    if categorical_features:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ]
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features)
            ]
        )

    def objective(trial):
        alpha = trial.suggest_float('alpha', 1e-3, 1.0, log=True)  # Narrowed range to avoid convergence issues
        l1_ratio = trial.suggest_float('l1_ratio', 0.1, 1.0)

        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=2000,  # Increased iterations
                tol=1e-3,       # Relaxed tolerance
                random_state=42
            ))
        ])

        try:
            score = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1).mean()
            return score
        except Exception as e:
            print(f"Trial failed: {e}")
            return -np.inf  # Return very low score for failed trials

    # Run optimization
    study = optuna.create_study(direction='maximize')

    # Suppress Optuna's verbose output
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(objective, n_trials=n_trials)

    print("Best ElasticNet Parameters:")
    print(study.best_params)
    print(f"Best cross-validated R²: {study.best_value:.4f}")

    # Train final model with best params
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