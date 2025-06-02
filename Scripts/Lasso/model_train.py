"""
Module for training lasso models on the COVID-19 dataset.
Includes data splitting, preprocessing, feature selection, model fitting, and evaluation.
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from .pipeline_utils import build_preprocessing_pipeline
import numpy as np


def train_lasso_model(df, config):
    """
    Trains a LassoCV model using a pipeline and state-based train/test split.

    Parameters:
        df (DataFrame): Full input DataFrame (features + target + state)
        target_col (str): Name of the target column

    Returns:
        model_pipeline (Pipeline): Trained model
        X_train, X_test, y_train, y_test: Train/test splits for future evaluation
    """
    # Split target and features
    X = df.drop(columns=[config["target_col"]])
    y = df[config["target_col"]]

    # Identify features
    categorical_features = ['state'] if 'state' in X.columns else []
    numerical_features = X.select_dtypes(include=['float64', 'int64']) \
                          .columns.difference(categorical_features).tolist()

    # Train/test split by state
    states = df['state'].unique()
    np.random.seed(config["seed"])
    np.random.shuffle(states)
    train_states = states[:int(config["train_size"] * len(states))]
    test_states = states[int(config["train_size"]* len(states)):]

    X_train = X[X['state'].isin(train_states)]
    y_train = y[X['state'].isin(train_states)]
    X_test = X[X['state'].isin(test_states)]
    y_test = y[X['state'].isin(test_states)]

    # Build pipeline
    preprocessor = build_preprocessing_pipeline(numerical_features, categorical_features)
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LassoCV(
            alphas=np.logspace(-4, 1, 50),
            max_iter=config["max_iter"],
            tol=1e-4,
            cv=config["cv"],
            random_state=config["seed"]
        ))
    ])

    # Train
    model_pipeline.fit(X_train, y_train)

    # Return everything for later evaluation
    return model_pipeline, X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()



