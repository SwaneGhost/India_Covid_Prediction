"""
Feature Selection Module using SelectKBest and f_regression.
"""

from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd
import numpy as np


def select_top_k_features(X, y, k='all', return_scores=False):
    """
    Selects top k numerical features based on univariate linear regression (f_regression).
    Categorical features are excluded from the selection process and retained as-is.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series or np.array): Target vector.
        k (int or 'all'): Number of features to select. If 'all', selects all.
        return_scores (bool): If True, returns a DataFrame of feature scores and p-values.

    Returns:
        selected_X (pd.DataFrame): Filtered DataFrame with top k features.
        selected_features (list): List of selected feature names.
        scores_df (pd.DataFrame, optional): Scores and p-values (if return_scores=True).
    """
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    if categorical_features:
        print(f"Categorical features found and excluded from selection: {categorical_features}")

    if not numerical_features:
        raise ValueError("No numerical features available for selection.")

    print(f"Selecting from {len(numerical_features)} numerical features")

    X_numerical = X[numerical_features]
    k = min(k, len(numerical_features)) if k != 'all' else len(numerical_features)
    print(f"Selecting top {k} features")

    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X_numerical, y)

    selected_mask = selector.get_support()
    selected_numerical_features = X_numerical.columns[selected_mask]

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
    Encodes categorical features and selects top k features using f_regression.
    Allows inclusion of categorical features by encoding them numerically.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series or np.array): Target vector.
        k (int): Number of features to select.
        return_scores (bool): If True, returns feature scores and p-values.

    Returns:
        selected_X (pd.DataFrame): Original DataFrame with only selected features.
        selected_features (list): List of selected feature names.
        scores_df (pd.DataFrame, optional): Scores and p-values (if return_scores=True).
    """
    from sklearn.preprocessing import LabelEncoder

    X_encoded = X.copy()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        encoders[col] = le

    k = min(k, len(X_encoded.columns))
    print(f"Selecting top {k} features from {len(X_encoded.columns)} total features")

    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X_encoded, y)

    selected_mask = selector.get_support()
    selected_features = X_encoded.columns[selected_mask].tolist()
    selected_X = X[selected_features]  # Return original values (not encoded)

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
