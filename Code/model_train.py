"""
Module for training ElasticNet models on the COVID-19 dataset.
Includes data splitting, preprocessing, optional feature selection, and model evaluation.
"""

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNetCV


def remove_highly_correlated_features(df, target_col='cum_positive_cases',
                                      target_corr_threshold=0.93,
                                      feature_corr_threshold=0.85):
    """
    Removes features highly correlated with the target or with each other.
    """
    df_clean = df.copy()
    drop_cols = ['dates', 'date']
    df_features = df_clean.drop(columns=drop_cols, errors='ignore')

    numeric_df = df_features.select_dtypes(include=[np.number])

    corr_with_target = numeric_df.corr()[target_col].drop(target_col)
    to_drop_target = corr_with_target[abs(corr_with_target) > target_corr_threshold].index.tolist()
    print(f"Removing features highly correlated with the target (> {target_corr_threshold}): {to_drop_target}")
    df_clean.drop(columns=to_drop_target, inplace=True)

    numeric_df = df_clean.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.drop(columns=[target_col], errors='ignore').corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_features = [col for col in upper_tri.columns if any(upper_tri[col] > feature_corr_threshold)]
    print(f"Removing features highly correlated with each other (> {feature_corr_threshold}): {to_drop_features}")
    df_clean.drop(columns=to_drop_features, inplace=True)

    print(f"Final shape after removing correlated features: {df_clean.shape}")
    return df_clean


def train_improved_elasticnet_model(df, split_type='random', use_feature_selection=True):
    """
    Trains an ElasticNetCV model with preprocessing and optional feature selection.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        split_type (str): One of 'random', 'by_state', or 'time'.
        use_feature_selection (bool): Whether to include SelectKBest before regression.

    Returns:
        model_pipeline: Trained scikit-learn pipeline.
        X_train_final, X_test_final: Feature sets.
        y_train, y_test: Targets.
    """
    print(f"Splitting method: {split_type}")

    target = 'cum_positive_cases'
    id_cols = ['dates', 'date']
    existing_id_cols = [col for col in id_cols if col in df.columns]
    X = df.drop(columns=[target] + existing_id_cols)
    y = df[target]

    categorical_features = ['state'] if 'state' in X.columns else []
    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.difference(categorical_features).tolist()

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

    if use_feature_selection:
        from sklearn.feature_selection import SelectKBest, f_regression
        feature_selector = SelectKBest(score_func=f_regression, k=min(20, len(numerical_features)))
        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_selection', feature_selector),
            ('regressor', ElasticNetCV(
                l1_ratio=[.1, .3, .5, .7, .9, .95, .99, 1],
                alphas=np.logspace(-4, 1, 50),
                max_iter=10000,
                cv=10,
                random_state=42
            ))
        ])
    else:
        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', ElasticNetCV(
                l1_ratio=[.1, .3, .5, .7, .9, .95, .99, 1],
                alphas=np.logspace(-4, 1, 50),
                max_iter=10000,
                cv=10,
                random_state=42
            ))
        ])

    if split_type == 'random':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    elif split_type == 'by_state':
        states = df['state'].unique()
        np.random.seed(42)
        np.random.shuffle(states)
        train_states = states[:int(0.8 * len(states))]
        X_train = X[df['state'].isin(train_states)]
        y_train = y[df['state'].isin(train_states)]
        X_test = X[~df['state'].isin(train_states)]
        y_test = y[~df['state'].isin(train_states)]
    elif split_type == 'time':
        df_sorted = df.sort_values(by='date' if 'date' in df.columns else df.index)
        split_index = int(0.8 * len(df_sorted))
        train_idx = df_sorted.index[:split_index]
        test_idx = df_sorted.index[split_index:]
        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]
    else:
        raise ValueError("split_type must be one of: 'random', 'by_state', 'time'")

    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_rmse = mean_squared_error(y_test, baseline_pred, squared=False)
    baseline_r2 = r2_score(y_test, baseline_pred)

    print(f"Test RMSE: {rmse:.2f} (Baseline: {baseline_rmse:.2f})")
    print(f"Test R²: {r2:.4f} (Baseline: {baseline_r2:.4f})")

    if hasattr(model_pipeline.named_steps['regressor'], 'alpha_'):
        print(f"Selected alpha: {model_pipeline.named_steps['regressor'].alpha_:.6f}")
        print(f"Selected l1_ratio: {model_pipeline.named_steps['regressor'].l1_ratio_:.2f}")

    print("Running cross-validation...")
    if split_type == 'by_state':
        cv = GroupKFold(n_splits=5)
        groups = X['state']
    elif split_type == 'time':
        cv = TimeSeriesSplit(n_splits=5)
        groups = None
    else:
        cv = 10
        groups = None

    scores = cross_val_score(model_pipeline, X, y, cv=cv, scoring='r2', groups=groups)
    print(f"Cross-validated R²: mean = {scores.mean():.4f}, std = {scores.std():.4f}")

    train_score = model_pipeline.score(X_train, y_train)
    overfitting_gap = train_score - r2
    print(f"Train R²: {train_score:.4f}")
    print(f"Overfitting gap: {overfitting_gap:.4f}")

    if overfitting_gap > 0.1:
        print("Model may be overfitting. Consider stronger regularization or simpler features.")

    return model_pipeline, X_train.copy(), X_test.copy(), y_train, y_test
