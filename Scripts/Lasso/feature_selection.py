"""
Feature Selection Module using SelectKBest and f_regression.
"""

from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd
import numpy as np


def select_top_k_features(X, y, config):
    """
    Selects the top k numerical features based on their relationship with the target.

    This uses f_regression to evaluate each numeric feature. Categorical features
    are excluded from scoring but included in the final output unchanged.

    Parameters:
        X (DataFrame): Input features.
        y (Series or array): Target variable.
        k (int or 'all'): Number of top features to select.
        return_scores (bool): Whether to return scores and p-values.

    Returns:
        selected_X (DataFrame): Filtered data with selected features.
        selected_features (list): Names of selected features.
        scores_df (DataFrame, optional): Feature scores and p-values if requested.
    """
    # Separate numeric and categorical features
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    if categorical_features:
        print(f"Categorical features found and excluded from selection: {categorical_features}")

    if not numerical_features:
        raise ValueError("No numerical features available for selection.")

    print(f"Selecting from {len(numerical_features)} numerical features")

    # Prepare numerical feature set
    X_numerical = X[numerical_features]
    config["k"] = min(config["k"], len(numerical_features)) if config["k"] != 'all' else len(numerical_features)

    # Apply SelectKBest with f_regression
    selector = SelectKBest(score_func=f_regression, k=config["k"])
    selector.fit(X_numerical, y)

    # Get selected features
    selected_mask = selector.get_support()
    selected_numerical_features = X_numerical.columns[selected_mask]

    # Combine selected numerical features with all categorical ones
    all_selected_features = selected_numerical_features.tolist() + categorical_features
    selected_X = X[all_selected_features]

    print(f"Selected {len(selected_numerical_features)} numerical and {len(categorical_features)} categorical features")
    return selected_X, all_selected_features

