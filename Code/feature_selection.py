"""
Feature Selection Module using SelectKBest and f_regression.
"""

from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd
import numpy as np


def select_top_k_features(X, y, k='all', return_scores=False):
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
    k = min(k, len(numerical_features)) if k != 'all' else len(numerical_features)
    print(f"Selecting top {k} features")

    # Apply SelectKBest with f_regression
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X_numerical, y)

    # Get selected features
    selected_mask = selector.get_support()
    selected_numerical_features = X_numerical.columns[selected_mask]

    # Combine selected numerical features with all categorical ones
    all_selected_features = selected_numerical_features.tolist() + categorical_features
    selected_X = X[all_selected_features]

    if return_scores:
        scores = selector.scores_
        pvalues = selector.pvalues_
        scores_df = pd.DataFrame({
            'feature': X_numerical.columns,
            'f_score': scores,
            'p_value': pvalues
        }).sort_values(by='f_score', ascending=False)

        print(f"Selected {len(selected_numerical_features)} numerical and {len(categorical_features)} categorical features")
        return selected_X, all_selected_features, scores_df

    print(f"Selected {len(selected_numerical_features)} numerical and {len(categorical_features)} categorical features")
    return selected_X, all_selected_features


def select_top_k_features_with_encoding(X, y, k=20, return_scores=False):
    """
    Selects top k features from both numerical and categorical columns.

    Categorical features are encoded using LabelEncoder before scoring.
    The original (non-encoded) values are returned in the output.

    Parameters:
        X (DataFrame): Input features.
        y (Series or array): Target variable.
        k (int): Number of top features to select.
        return_scores (bool): Whether to return scores and p-values.

    Returns:
        selected_X (DataFrame): Filtered data with selected original features.
        selected_features (list): Names of selected features.
        scores_df (DataFrame, optional): Feature scores and p-values if requested.
    """
    from sklearn.preprocessing import LabelEncoder

    # Copy original data for encoding
    X_encoded = X.copy()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Encode each categorical column
    encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        encoders[col] = le

    # Make sure k does not exceed total number of columns
    k = min(k, len(X_encoded.columns))
    print(f"Selecting top {k} features from {len(X_encoded.columns)} total features")

    # Apply SelectKBest with f_regression
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X_encoded, y)

    # Get selected features and return their original values
    selected_mask = selector.get_support()
    selected_features = X_encoded.columns[selected_mask].tolist()
    selected_X = X[selected_features]

    if return_scores:
        scores = selector.scores_
        pvalues = selector.pvalues_
        scores_df = pd.DataFrame({
            'feature': X_encoded.columns,
            'f_score': scores,
            'p_value': pvalues
        }).sort_values(by='f_score', ascending=False)

        print(f"Selected {len(selected_features)} features")
        return selected_X, selected_features, scores_df

    print(f"Selected {len(selected_features)} features")
    return selected_X, selected_features
