import numpy as np

def remove_highly_correlated_features(df, target_col='cum_positive_cases',
                                      target_corr_threshold=0.85,
                                      feature_corr_threshold=0.8):
    """
    Removes features that are:
    - Too correlated with the target (can leak future info),
    - Too correlated with each other (adds redundancy).

    Keeps the data clean and prevents overfitting.
    """

    df_clean = df.copy()

    # Drop non-feature columns if present
    drop_cols = ['dates', 'date','cum_tests']
    df_features = df_clean.drop(columns=[col for col in drop_cols if col in df_clean.columns], errors='ignore')

    # Keep only numeric features
    numeric_df = df_features.select_dtypes(include=[np.number])

    # Remove features that correlate too much with the target
    if target_col in numeric_df.columns:
        corr_with_target = numeric_df.corr()[target_col].drop(target_col)
        to_drop_target = corr_with_target[abs(corr_with_target) > target_corr_threshold].index.tolist()
        df_clean.drop(columns=to_drop_target, inplace=True)
        print(f"Removed features too correlated with target ({target_corr_threshold}+):", to_drop_target)

    # Remove features highly correlated with each other
    numeric_df = df_clean.select_dtypes(include=[np.number])
    feature_cols = numeric_df.drop(columns=[target_col], errors='ignore')

    if len(feature_cols.columns) > 1:
        corr_matrix = feature_cols.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop_features = [
            col for col in upper_triangle.columns
            if any(upper_triangle[col] > feature_corr_threshold)
        ]
        df_clean.drop(columns=to_drop_features, inplace=True)
        print(f"Removed highly correlated features ({feature_corr_threshold}+):", to_drop_features)

    print("Final shape after correlation filtering:", df_clean.shape)
    return df_clean


def feature_engineering(df):
    """
    Adds useful engineered features that help model performance.

    These are conservative, interpretable transformations that reduce noise
    and improve signal — without overcomplicating the dataset.
    """

    df_fe = df.copy()

    # Normalize testing and case counts
    if 'daily_tests' in df_fe.columns and 'population' in df_fe.columns:
        df_fe['tests_per_capita'] = df_fe['daily_tests'] / (df_fe['population'] + 1e-5)

    if 'daily_positive_cases' in df_fe.columns and 'population' in df_fe.columns:
        df_fe['cases_per_capita'] = df_fe['daily_positive_cases'] / (df_fe['population'] + 1e-5)

    # Healthcare beds: sum across types and normalize
    bed_cols = ['rural_beds', 'urban_beds', 'public_beds']
    existing_beds = [col for col in bed_cols if col in df_fe.columns]

    if existing_beds and 'population' in df_fe.columns:
        df_fe['total_beds'] = df_fe[existing_beds].sum(axis=1)
        df_fe['beds_per_1000'] = df_fe['total_beds'] / (df_fe['population'] / 1000 + 1e-5)

    # Testing efficiency
    if 'daily_positive_cases' in df_fe.columns and 'daily_tests' in df_fe.columns:
        df_fe['test_efficiency'] = df_fe['daily_positive_cases'] / (df_fe['daily_tests'] + 1e-5)

    # Demographic interaction: density * female-to-male ratio
    if 'density' in df_fe.columns and 'Female to Male ratio' in df_fe.columns:
        df_fe['elderly_density'] = df_fe['density'] * df_fe['Female to Male ratio']


    print("Feature engineering completed.")
    return df_fe
