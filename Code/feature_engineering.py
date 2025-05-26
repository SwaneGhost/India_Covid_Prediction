"""
Module for feature engineering on the COVID-19 dataset.
"""

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
#
# def engineer_features(df):
#     """
#     Engineer features for COVID-19 prediction model.
#
#     Args:
#         df (pd.DataFrame): The merged dataset containing COVID-19 data.
#
#     Returns:
#         tuple: (X_engineered, y, existing_demo_socio_cols) where:
#             - X_engineered is the dataframe with engineered features
#             - y is the target variable
#             - existing_demo_socio_cols is a list of demographic and socioeconomic columns
#     """
#     # Define your demographic and socioeconomic columns
#     demographic_socioeconomic_cols = [
#         'population', 'Male literacy rate %', 'Female literacy rate %',
#         'Average literacy rate %', 'GDP', 'per capita in', 'Female to Male ratio',
#         'Hindu', 'Muslim', 'Christian', 'Sikhs', 'Buddhist', 'Others', 'density',
#         'primary_health_centers', 'community_health_centers', 'sub_district_hospitals',
#         'district_hospitals', 'public_health_facilities', 'public_beds',
#         'rural_hospitals', 'rural_beds', 'urban_hospitals', 'urban_beds',
#         'state'
#     ]
#
#     print("\nTotal demographic and socioeconomic columns:", len(demographic_socioeconomic_cols))
#     print(demographic_socioeconomic_cols)
#
#     # Check which demographic and socioeconomic columns exist in the dataset
#     existing_demo_socio_cols = [col for col in demographic_socioeconomic_cols if col in df.columns]
#     print(f"\nExisting demographic and socioeconomic columns: {len(existing_demo_socio_cols)}")
#     print(existing_demo_socio_cols)
#
#     # Check for missing values in demographic and socioeconomic columns
#     print("\nMissing values in demographic and socioeconomic columns:")
#     missing_values = df[existing_demo_socio_cols].isnull().sum()
#     print(missing_values[missing_values > 0])
#
#     # Define the target variable
#     y = df['cum_positive_cases']
#
#     # Get the latest data for each state for feature engineering
#     latest_date = df['date'].max()
#     latest_data = df[df['date'] == latest_date].copy()
#
#     # Apply feature engineering to the latest data
#     X_engineered = engineer_demographic_features(latest_data, existing_demo_socio_cols)
#
#     # Visualize correlations of new engineered features with target
#     numeric_engineered = X_engineered.select_dtypes(include=['number']).columns
#
#     # Merge with the target for correlation calculation
#     correlation_data = pd.concat([X_engineered[numeric_engineered],
#                                  pd.DataFrame({'cum_positive_cases': y.loc[latest_data.index]})],
#                                 axis=1)
#
#     correlation = correlation_data.corr()['cum_positive_cases'].sort_values(ascending=False)
#     print("\nTop 15 correlations with cumulative positive cases (engineered features):")
#     print(correlation.head(15))
#
#     # Visualize top engineered feature correlations
#     plt.figure(figsize=(14, 10))
#     top_corr = correlation.head(15)
#     correlation_df = pd.DataFrame(top_corr).reset_index()
#     correlation_df.columns = ['Feature', 'Correlation']
#     sns.barplot(x='Correlation', y='Feature', data=correlation_df)
#     plt.title('Top 15 Engineered Features: Correlation with Cumulative Positive Cases')
#     plt.axvline(x=0, color='r', linestyle='--')
#     plt.tight_layout()
#     plt.show()
#
#     return X_engineered, y.loc[latest_data.index], existing_demo_socio_cols
#
# def engineer_demographic_features(df, demographic_cols):
#     """
#     Comprehensive feature engineering for demographic and socioeconomic variables.
#
#     Args:
#         df (pd.DataFrame): DataFrame containing demographic and socioeconomic data.
#         demographic_cols (list): List of demographic and socioeconomic column names.
#
#     Returns:
#         pd.DataFrame: DataFrame with engineered features.
#     """
#     # Create a copy to avoid modifying the original
#     X = df[demographic_cols].copy()
#
#     print("\n=== FEATURE ENGINEERING ===")
#
#     print("1. Creating per capita features...")
#     # Healthcare per capita features
#     if all(col in X.columns for col in ['population', 'public_health_facilities']):
#         X['health_facilities_per_capita'] = X['public_health_facilities'] / (X['population'] + 0.001)
#
#     if all(col in X.columns for col in ['population', 'public_beds']):
#         X['beds_per_capita'] = X['public_beds'] / (X['population'] + 0.001)
#
#     if all(col in X.columns for col in ['population', 'primary_health_centers']):
#         X['primary_centers_per_capita'] = X['primary_health_centers'] / (X['population'] + 0.001)
#
#     # Economic per capita features
#     if all(col in X.columns for col in ['GDP', 'population']):
#         X['GDP_per_capita'] = X['GDP'] / (X['population'] + 0.001)
#
#     print("2. Creating ratio and proportion features...")
#     # Urban-Rural ratios
#     if all(col in X.columns for col in ['urban_hospitals', 'rural_hospitals']):
#         X['urban_rural_hospital_ratio'] = X['urban_hospitals'] / (X['rural_hospitals'] + 0.001)
#         X['urbanization_index'] = X['urban_hospitals'] / (X['urban_hospitals'] + X['rural_hospitals'] + 0.001)
#
#     if all(col in X.columns for col in ['urban_beds', 'rural_beds']):
#         X['urban_rural_beds_ratio'] = X['urban_beds'] / (X['rural_beds'] + 0.001)
#         X['total_beds'] = X['urban_beds'] + X['rural_beds']
#
#     # Healthcare facility ratios
#     if all(col in X.columns for col in ['public_beds', 'public_health_facilities']):
#         X['beds_per_facility'] = X['public_beds'] / (X['public_health_facilities'] + 0.001)
#
#     # Gender-related features
#     if all(col in X.columns for col in ['Male literacy rate %', 'Female literacy rate %']):
#         X['literacy_gender_gap'] = X['Male literacy rate %'] - X['Female literacy rate %']
#         X['literacy_gender_ratio'] = X['Female literacy rate %'] / (X['Male literacy rate %'] + 0.001)
#
#     print("3. Creating diversity and inequality indices...")
#     # Religious diversity (Herfindahl-Hirschman Index)
#     religion_cols = ['Hindu', 'Muslim', 'Christian', 'Sikhs', 'Buddhist', 'Others']
#     religion_cols = [col for col in religion_cols if col in X.columns]
#
#     if len(religion_cols) >= 2:  # Need at least two religion columns for diversity index
#         # Normalize to sum to 1
#         religion_sum = X[religion_cols].sum(axis=1)
#         for col in religion_cols:
#             X[f'{col}_norm'] = X[col] / (religion_sum + 0.001)
#
#         # Calculate diversity index (1 - HHI)
#         X['religious_diversity'] = 1 - ((X[[f'{col}_norm' for col in religion_cols]]**2).sum(axis=1))
#
#         # Drop the normalized columns as they're not needed anymore
#         X = X.drop(columns=[f'{col}_norm' for col in religion_cols])
#
#     # Healthcare diversity
#     health_cols = [col for col in X.columns if any(term in col for term in ['health', 'hospital', 'beds'])]
#     if len(health_cols) >= 3:  # Only if we have enough healthcare features
#         # Create a healthcare infrastructure score
#         X_health = X[health_cols].fillna(X[health_cols].median())
#         scaler = StandardScaler()
#         health_scaled = scaler.fit_transform(X_health)
#         X['healthcare_infrastructure_score'] = np.mean(health_scaled, axis=1)
#
#         # Calculate variance in healthcare metrics as a measure of healthcare inequality
#         X['healthcare_inequality'] = np.std(health_scaled, axis=1)
#
#     print("4. Applying non-linear transformations...")
#     # Log transformations for skewed features
#     numeric_cols = X.select_dtypes(include=['number']).columns
#     skewed_features = []
#     for col in numeric_cols:
#         if col in X.columns and X[col].skew() > 1.5:  # Check for highly skewed features
#             skewed_features.append(col)
#
#     print(f"   Found {len(skewed_features)} highly skewed features")
#     for col in skewed_features:
#         min_val = X[col].min()
#         # Ensure all values are positive before log transform
#         if min_val <= 0:
#             X[f'{col}_log'] = np.log(X[col] - min_val + 1)
#         else:
#             X[f'{col}_log'] = np.log(X[col])
#
#     print("5. Creating interaction features...")
#     # Create meaningful interaction terms
#     interaction_pairs = [
#         # Healthcare and population density
#         ('density', 'public_health_facilities'),
#         ('density', 'public_beds'),
#         # Literacy and healthcare
#         ('Average literacy rate %', 'public_health_facilities'),
#         # Economic status and healthcare
#         ('GDP', 'public_beds'),
#         ('GDP', 'public_health_facilities')
#     ]
#
#     for col1, col2 in interaction_pairs:
#         if col1 in X.columns and col2 in X.columns:
#             X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
#
#     print("6. Creating cluster-based features...")
#     # Cluster regions based on socioeconomic profiles
#     numeric_cols = X.select_dtypes(include=['number']).columns
#     if len(numeric_cols) >= 3:  # Need at least a few features for meaningful clustering
#         try:
#             # Fill missing values for clustering
#             X_cluster = X[numeric_cols].fillna(X[numeric_cols].median())
#
#             # Standardize the data
#             scaler = StandardScaler()
#             X_scaled = scaler.fit_transform(X_cluster)
#
#             # Apply KMeans clustering
#             kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
#             X['socioeconomic_cluster'] = kmeans.fit_predict(X_scaled)
#
#             # Create one-hot encoded cluster features
#             cluster_dummies = pd.get_dummies(X['socioeconomic_cluster'], prefix='cluster')
#             X = pd.concat([X, cluster_dummies], axis=1)
#         except Exception as e:
#             print(f"Error in clustering: {str(e)}")
#
#     print("7. Creating PCA-based features...")
#     # Apply PCA for dimensionality reduction
#     if len(numeric_cols) >= 5:  # Only if we have enough features
#         try:
#             # Fill missing values for PCA
#             X_pca = X[numeric_cols].fillna(X[numeric_cols].median())
#
#             # Standardize data for PCA
#             scaler = StandardScaler()
#             X_scaled = scaler.fit_transform(X_pca)
#
#             # Apply PCA
#             pca = PCA(n_components=0.95)  # Capture 95% of variance
#             pca_components = pca.fit_transform(X_scaled)
#
#             # Add PCA components as features
#             for i in range(pca_components.shape[1]):
#                 X[f'pca_component_{i+1}'] = pca_components[:, i]
#
#             print(f"   Added {pca_components.shape[1]} PCA components")
#         except Exception as e:
#             print(f"Error in PCA: {str(e)}")
#
#     print(f"\nFeature engineering complete! Created {X.shape[1]} features from {len(demographic_cols)} original features.")
#     return X





##########################################################################################################

import numpy as np
def remove_data_leakage_features(df, target_col='cum_positive_cases'):
    """Remove features that cause data leakage"""
    df_clean = df.copy()

    # Features that are directly derived from or highly predictive of target
    leakage_features = [
        'cum_recovered',  # Directly related to cases
        'cum_deceased',  # Directly related to cases
        'cum_tests',  # Often highly correlated with cases
        'cases_to_tests_ratio',  # Uses cum_positive_cases in calculation
        'deaths_to_cases_ratio',  # Uses cum_positive_cases in calculation
        'recovery_rate',  # Uses cum_positive_cases in calculation
    ]

    # Remove leakage features that exist in the dataframe
    existing_leakage = [col for col in leakage_features if col in df_clean.columns]
    df_clean = df_clean.drop(columns=existing_leakage)

    print(f"🚫 Removed potential data leakage features: {existing_leakage}")
    return df_clean


def remove_highly_correlated_features(df, target_col='cum_positive_cases',
                                      target_corr_threshold=0.85,  # Lowered threshold
                                      feature_corr_threshold=0.8):  # Lowered threshold
    df_clean = df.copy()

    # Remove identifiers
    drop_cols = ['dates', 'date']
    existing_drop_cols = [col for col in drop_cols if col in df_clean.columns]
    df_features = df_clean.drop(columns=existing_drop_cols)

    # Keep only numeric features for correlation
    numeric_df = df_features.select_dtypes(include=[np.number])

    # 1. Correlation with target
    if target_col in numeric_df.columns:
        corr_with_target = numeric_df.corr()[target_col].drop(target_col)
        to_drop_target = corr_with_target[abs(corr_with_target) > target_corr_threshold].index.tolist()

        print(f"📉 Features too correlated with target (> {target_corr_threshold}):")
        print(to_drop_target)

        df_clean = df_clean.drop(columns=to_drop_target)

    # 2. Correlation between features
    numeric_df = df_clean.select_dtypes(include=[np.number])
    if target_col in numeric_df.columns:
        feature_cols = numeric_df.drop(columns=[target_col])
    else:
        feature_cols = numeric_df

    if len(feature_cols.columns) > 1:
        corr_matrix = feature_cols.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop_features = [column for column in upper_tri.columns if any(upper_tri[column] > feature_corr_threshold)]

        print(f"🔁 Features highly correlated with others (> {feature_corr_threshold}):")
        print(to_drop_features)

        df_clean = df_clean.drop(columns=to_drop_features)

    print(f"✅ Final shape after removing correlated features: {df_clean.shape}")

    return df_clean

def improved_feature_engineering(df):
    """More conservative feature engineering to reduce overfitting"""
    df_fe = df.copy()

    # Only create features that are theoretically sound and less prone to overfitting

    # 1. Per capita metrics (normalized by population)
    if 'daily_tests' in df_fe.columns and 'population' in df_fe.columns:
        df_fe['tests_per_capita'] = df_fe['daily_tests'] / (df_fe['population'] + 1e-5)

    if 'daily_positive_cases' in df_fe.columns and 'population' in df_fe.columns:
        df_fe['cases_per_capita'] = df_fe['daily_positive_cases'] / (df_fe['population'] + 1e-5)

    # 2. Healthcare capacity (less likely to cause overfitting)
    bed_cols = ['rural_beds', 'urban_beds', 'public_beds']
    existing_bed_cols = [col for col in bed_cols if col in df_fe.columns]
    if existing_bed_cols and 'population' in df_fe.columns:
        df_fe['total_beds'] = df_fe[existing_bed_cols].sum(axis=1)
        df_fe['beds_per_1000'] = df_fe['total_beds'] / (df_fe['population'] / 1000 + 1e-5)

    # 3. Simple testing efficiency (but avoid using cumulative cases)
    if 'daily_positive_cases' in df_fe.columns and 'daily_tests' in df_fe.columns:
        df_fe['test_efficiency'] = df_fe['daily_positive_cases'] / (df_fe['daily_tests'] + 1e-5)

    # 4. Demographic indicators
    if 'density' in df_fe.columns and 'Female to Male ratio' in df_fe.columns:
        df_fe['elderly_density'] = df_fe['density'] * df_fe['Female to Male ratio']

    # Remove features that might cause leakage
    df_fe = remove_data_leakage_features(df_fe)

    print("✅ Conservative feature engineering completed.")

    return df_fe