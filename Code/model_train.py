"""
Module for training lasso models on the COVID-19 dataset.
Includes data splitting, preprocessing, feature selection, model fitting, and evaluation.
"""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import  LassoCV


def remove_highly_correlated_features(df, target_col='cum_positive_cases',
                                      target_corr_threshold=0.93,
                                      feature_corr_threshold=0.85):
    """
    Removes columns that are too correlated with the target or with each other.
    Helps reduce data leakage and multicollinearity.
    """

    df_clean = df.copy()

    # Drop identifier columns if they exist
    drop_cols = ['dates', 'date']
    df_features = df_clean.drop(columns=drop_cols, errors='ignore')

    # Focus only on numeric columns
    numeric_df = df_features.select_dtypes(include=[np.number])

    # Remove features too strongly correlated with the target
    corr_with_target = numeric_df.corr()[target_col].drop(target_col)
    to_drop_target = corr_with_target[abs(corr_with_target) > target_corr_threshold].index.tolist()
    print(f"Removing features highly correlated with the target (> {target_corr_threshold}): {to_drop_target}")
    df_clean.drop(columns=to_drop_target, inplace=True)

    # Remove features too strongly correlated with other features
    numeric_df = df_clean.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.drop(columns=[target_col], errors='ignore').corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_features = [col for col in upper_tri.columns if any(upper_tri[col] > feature_corr_threshold)]
    print(f"Removing features highly correlated with each other (> {feature_corr_threshold}): {to_drop_features}")
    df_clean.drop(columns=to_drop_features, inplace=True)

    print(f"Final shape after removing correlated features: {df_clean.shape}")
    return df_clean


def train_lasso_model(df, split_type='by_state', custom_target=None):
    """
    Trains a lasso model using a pipeline with preprocessing and feature selection.
    Supports different data splitting methods and prints evaluation metrics.

    Parameters:
        df (DataFrame): Full input data with features and target.
        split_type (str): How to split the data: 'by_state'.
        custom_target (Series, optional): Use a transformed target instead of the default.

    Returns:
        model_pipeline (Pipeline): Trained pipeline.
        X_train, X_test (DataFrame): Training and testing features.
        y_train, y_test (Series): Training and testing targets.
    """
    print(f"Splitting method: {split_type}")

    target = 'cum_positive_cases'
    id_cols = ['dates', 'date']
    existing_id_cols = [col for col in id_cols if col in df.columns]
    X = df.drop(columns=[target] + existing_id_cols)
    y = custom_target if custom_target is not None else df[target]

    # Identify categorical and numerical features
    categorical_features = ['state'] if 'state' in X.columns else []
    numerical_features = X.select_dtypes(include=['float64', 'int64']) \
                          .columns.difference(categorical_features).tolist()

    # Remove low-variance numerical features
    vt = VarianceThreshold(threshold=1e-4)
    X_num_filtered = vt.fit_transform(X[numerical_features])
    retained_numerical = np.array(numerical_features)[vt.get_support()].tolist()

    # Select top numerical features using f_regression
    selector = SelectKBest(score_func=f_regression, k=min(20, len(retained_numerical)))
    X_kbest_filtered = selector.fit_transform(X[retained_numerical], y)
    selected_numerical = np.array(retained_numerical)[selector.get_support()].tolist()

    # Build preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ]), selected_numerical),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ] if categorical_features else [
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ]), selected_numerical)
        ]
    )

    # Build model pipeline with lasso
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LassoCV(
            alphas=np.logspace(-4, 1, 50),
            max_iter=20000,
            tol=1e-4,
            cv=10,
            random_state=42
        ))
    ])

    # Choose how to split the data
    if split_type == 'by_state':
        states = df['state'].unique()
        np.random.seed(42)
        np.random.shuffle(states)
        train_states = states[:int(0.8 * len(states))]
        X_train = X[df['state'].isin(train_states)]
        y_train = y[df['state'].isin(train_states)]
        X_test = X[~df['state'].isin(train_states)]
        y_test = y[~df['state'].isin(train_states)]
    else:
        raise ValueError("split_type must be one of: 'by_state'")

    # Fit the pipeline on training data
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)

    # Evaluate model on test set
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R²: {r2:.4f}")

    # Print best alpha  found during CV
    if hasattr(model_pipeline.named_steps['regressor'], 'alpha_'):
        print(f"Selected alpha: {model_pipeline.named_steps['regressor'].alpha_:.6f}")

    # Run cross-validation to check model performance
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

    # Compare train vs test R²
    train_score = model_pipeline.score(X_train, y_train)
    overfitting_gap = train_score - r2
    print(f"Train R²: {train_score:.4f}")
    print(f"Overfitting gap: {overfitting_gap:.4f}")

    return model_pipeline, X_train.copy(), X_test.copy(), y_train, y_test
