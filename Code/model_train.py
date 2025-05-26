"""
Module for training machine learning models on the COVID-19 dataset.
"""

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score


def remove_highly_correlated_features(df, target_col='cum_positive_cases',
                                       target_corr_threshold=0.93,
                                       feature_corr_threshold=0.85):
    df_clean = df.copy()

    # Remove identifiers
    drop_cols = ['dates', 'date']
    df_features = df_clean.drop(columns=drop_cols)

    # Keep only numeric features for correlation
    numeric_df = df_features.select_dtypes(include=[np.number])

    # 1. Correlation with target
    corr_with_target = numeric_df.corr()[target_col].drop(target_col)
    to_drop_target = corr_with_target[abs(corr_with_target) > target_corr_threshold].index.tolist()

    print(f"📉 Features too correlated with target (> {target_corr_threshold}):")
    print(to_drop_target)

    df_clean.drop(columns=to_drop_target, inplace=True)

    # 2. Correlation between features
    numeric_df = df_clean.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.drop(columns=[target_col]).corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop_features = [column for column in upper_tri.columns if any(upper_tri[column] > feature_corr_threshold)]

    print(f"🔁 Features highly correlated with others (> {feature_corr_threshold}):")
    print(to_drop_features)

    df_clean.drop(columns=to_drop_features, inplace=True)

    print(f"✅ Final shape after removing correlated features: {df_clean.shape}")

    return df_clean


from sklearn.linear_model import ElasticNetCV

def train_improved_elasticnet_model(df, split_type='random', use_feature_selection=True):
    """Improved ElasticNet model with feature cleaning and conservative engineering"""

    df = remove_highly_correlated_features(df)
    print(f"🔧 Splitting type: {split_type}")

    # Define target and features
    target = 'cum_positive_cases'
    id_cols = ['dates', 'date']
    existing_id_cols = [col for col in id_cols if col in df.columns]
    X = df.drop(columns=[target] + existing_id_cols)
    y = df[target]

    categorical_features = ['state'] if 'state' in X.columns else []
    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.difference(categorical_features).tolist()

    # Preprocessing
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

    # Feature selection + ElasticNetCV
    if use_feature_selection:
        from sklearn.feature_selection import SelectKBest, f_regression
        feature_selector = SelectKBest(score_func=f_regression, k=min(20, len(numerical_features)))

        model = Pipeline(steps=[
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
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', ElasticNetCV(
                l1_ratio=[.1, .3, .5, .7, .9, .95, .99, 1],
                alphas=np.logspace(-4, 1, 50),
                max_iter=10000,
                cv=10,
                random_state=42
            ))
        ])

    # === Splitting ===
    if split_type == 'random':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)

    elif split_type == 'by_state':
        unique_states = df['state'].unique()
        np.random.seed(42)
        np.random.shuffle(unique_states)
        train_states = unique_states[:int(0.8 * len(unique_states))]
        X_train = X[df['state'].isin(train_states)]
        y_train = y[df['state'].isin(train_states)]
        X_test = X[~df['state'].isin(train_states)]
        y_test = y[~df['state'].isin(train_states)]
        model.fit(X_train, y_train)

    elif split_type == 'time':
        df_sorted = df.sort_values(by='date' if 'date' in df.columns else df.index)
        split_index = int(0.8 * len(df_sorted))
        train_idx = df_sorted.index[:split_index]
        test_idx = df_sorted.index[split_index:]
        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]
        model.fit(X_train, y_train)

    else:
        raise ValueError("Invalid split_type. Choose from 'random', 'by_state', or 'time'.")

    # === Evaluation ===
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_rmse = mean_squared_error(y_test, baseline_pred, squared=False)
    baseline_r2 = r2_score(y_test, baseline_pred)

    print(f"📈 RMSE: {rmse:.2f} (Baseline: {baseline_rmse:.2f})")
    print(f"📊 R² Score: {r2:.4f} (Baseline: {baseline_r2:.4f})")

    if hasattr(model.named_steps['regressor'], 'alpha_'):
        print(f"🎯 Selected ElasticNet alpha: {model.named_steps['regressor'].alpha_:.6f}")
        print(f"🔗 Selected l1_ratio: {model.named_steps['regressor'].l1_ratio_:.2f}")

    # === Cross-validation ===
    print("⏳ Performing cross-validation...")
    if split_type == 'by_state':
        cv = GroupKFold(n_splits=5)
        groups = X['state']
    elif split_type == 'time':
        cv = TimeSeriesSplit(n_splits=5)
        groups = None
    else:
        cv = 10
        groups = None

    scores = cross_val_score(model, X, y, cv=cv, scoring='r2', groups=groups)
    print(f"✅ Cross-validated R²: Mean = {scores.mean():.4f}, Std = {scores.std():.4f}")

    # Overfitting check
    train_score = model.score(X_train, y_train)
    overfitting_gap = train_score - r2
    print(f"🔍 Training R²: {train_score:.4f}")
    print(f"🔍 Test R²: {r2:.4f}")
    print(f"⚠ Overfitting gap: {overfitting_gap:.4f}")

    if overfitting_gap > 0.1:
        print("🚨 Overfitting detected. Try more regularization or simpler features.")

    return model



