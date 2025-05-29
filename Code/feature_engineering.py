import numpy as np

def remove_data_leakage_features(df, target_col='cum_positive_cases'):
    """
    Removes columns that are likely to leak information from the target variable.
    These features are typically derived from or highly correlated with the target.
    """
    df_clean = df.copy()

    leakage_features = [
        'cum_recovered',
        'cum_deceased',
        'cum_tests',
        'cases_to_tests_ratio',
        'deaths_to_cases_ratio',
        'recovery_rate',
    ]

    existing_leakage = [col for col in leakage_features if col in df_clean.columns]
    df_clean.drop(columns=existing_leakage, inplace=True)

    print("Removed data leakage features:", existing_leakage)
    return df_clean


def remove_highly_correlated_features(df, target_col='cum_positive_cases',
                                      target_corr_threshold=0.85,
                                      feature_corr_threshold=0.8):
    """
    Removes features that are either:
    1. Highly correlated with the target (which can cause leakage),
    2. Highly correlated with each other (to reduce redundancy).
    """
    df_clean = df.copy()

    drop_cols = ['dates', 'date']
    df_features = df_clean.drop(columns=[col for col in drop_cols if col in df_clean.columns], errors='ignore')
    numeric_df = df_features.select_dtypes(include=[np.number])

    # Remove features correlated with the target
    if target_col in numeric_df.columns:
        corr_with_target = numeric_df.corr()[target_col].drop(target_col)
        to_drop_target = corr_with_target[abs(corr_with_target) > target_corr_threshold].index.tolist()
        df_clean.drop(columns=to_drop_target, inplace=True)
        print(f"Removed features too correlated with target ({target_corr_threshold}+):", to_drop_target)

    # Remove features that are highly correlated with each other
    numeric_df = df_clean.select_dtypes(include=[np.number])
    feature_cols = numeric_df.drop(columns=[target_col], errors='ignore')

    if len(feature_cols.columns) > 1:
        corr_matrix = feature_cols.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_features = [col for col in upper_triangle.columns if any(upper_triangle[col] > feature_corr_threshold)]
        df_clean.drop(columns=to_drop_features, inplace=True)
        print(f"Removed highly correlated features ({feature_corr_threshold}+):", to_drop_features)

    print("Final shape after correlation filtering:", df_clean.shape)
    return df_clean


def improved_feature_engineering(df):
    """
    Applies a set of conservative feature engineering transformations.
    These transformations aim to improve model performance without introducing noise or overfitting.
    """
    df_fe = df.copy()

    # Create normalized testing and case features
    if 'daily_tests' in df_fe.columns and 'population' in df_fe.columns:
        df_fe['tests_per_capita'] = df_fe['daily_tests'] / (df_fe['population'] + 1e-5)

    if 'daily_positive_cases' in df_fe.columns and 'population' in df_fe.columns:
        df_fe['cases_per_capita'] = df_fe['daily_positive_cases'] / (df_fe['population'] + 1e-5)

    # Aggregate healthcare bed data and normalize
    bed_cols = ['rural_beds', 'urban_beds', 'public_beds']
    existing_beds = [col for col in bed_cols if col in df_fe.columns]
    if existing_beds and 'population' in df_fe.columns:
        df_fe['total_beds'] = df_fe[existing_beds].sum(axis=1)
        df_fe['beds_per_1000'] = df_fe['total_beds'] / (df_fe['population'] / 1000 + 1e-5)

    # Basic testing efficiency
    if 'daily_positive_cases' in df_fe.columns and 'daily_tests' in df_fe.columns:
        df_fe['test_efficiency'] = df_fe['daily_positive_cases'] / (df_fe['daily_tests'] + 1e-5)

    # Simple demographic interaction term
    if 'density' in df_fe.columns and 'Female to Male ratio' in df_fe.columns:
        df_fe['elderly_density'] = df_fe['density'] * df_fe['Female to Male ratio']

    # Clean up potential leakage
    df_fe = remove_data_leakage_features(df_fe)

    print("Feature engineering completed.")
    return df_fe
