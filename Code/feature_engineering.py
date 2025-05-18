"""
Module for feature engineering on the COVID-19 dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def engineer_features(df):
    """
    Engineer features for COVID-19 prediction model.
    
    Args:
        df (pd.DataFrame): The merged dataset containing COVID-19 data.
        
    Returns:
        tuple: (X_engineered, y, existing_demo_socio_cols) where:
            - X_engineered is the dataframe with engineered features
            - y is the target variable
            - existing_demo_socio_cols is a list of demographic and socioeconomic columns
    """
    # Define your demographic and socioeconomic columns
    demographic_socioeconomic_cols = [
        'population', 'Male literacy rate %', 'Female literacy rate %',
        'Average literacy rate %', 'GDP', 'per capita in', 'Female to Male ratio',
        'Hindu', 'Muslim', 'Christian', 'Sikhs', 'Buddhist', 'Others', 'density',
        'primary_health_centers', 'community_health_centers', 'sub_district_hospitals',
        'district_hospitals', 'public_health_facilities', 'public_beds',
        'rural_hospitals', 'rural_beds', 'urban_hospitals', 'urban_beds',
        'state'
    ]

    print("\nTotal demographic and socioeconomic columns:", len(demographic_socioeconomic_cols))
    print(demographic_socioeconomic_cols)

    # Check which demographic and socioeconomic columns exist in the dataset
    existing_demo_socio_cols = [col for col in demographic_socioeconomic_cols if col in df.columns]
    print(f"\nExisting demographic and socioeconomic columns: {len(existing_demo_socio_cols)}")
    print(existing_demo_socio_cols)

    # Check for missing values in demographic and socioeconomic columns
    print("\nMissing values in demographic and socioeconomic columns:")
    missing_values = df[existing_demo_socio_cols].isnull().sum()
    print(missing_values[missing_values > 0])

    # Define the target variable
    y = df['cum_positive_cases']

    # Get the latest data for each state for feature engineering
    latest_date = df['dates'].max()
    latest_data = df[df['dates'] == latest_date].copy()

    # Apply feature engineering to the latest data
    X_engineered = engineer_demographic_features(latest_data, existing_demo_socio_cols)

    # Visualize correlations of new engineered features with target
    numeric_engineered = X_engineered.select_dtypes(include=['number']).columns
    
    # Merge with the target for correlation calculation
    correlation_data = pd.concat([X_engineered[numeric_engineered], 
                                 pd.DataFrame({'cum_positive_cases': y.loc[latest_data.index]})], 
                                axis=1)
    
    correlation = correlation_data.corr()['cum_positive_cases'].sort_values(ascending=False)
    print("\nTop 15 correlations with cumulative positive cases (engineered features):")
    print(correlation.head(15))

    # Visualize top engineered feature correlations
    plt.figure(figsize=(14, 10))
    top_corr = correlation.head(15)
    correlation_df = pd.DataFrame(top_corr).reset_index()
    correlation_df.columns = ['Feature', 'Correlation']
    sns.barplot(x='Correlation', y='Feature', data=correlation_df)
    plt.title('Top 15 Engineered Features: Correlation with Cumulative Positive Cases')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.show()

    return X_engineered, y.loc[latest_data.index], existing_demo_socio_cols

def engineer_demographic_features(df, demographic_cols):
    """
    Comprehensive feature engineering for demographic and socioeconomic variables.
    
    Args:
        df (pd.DataFrame): DataFrame containing demographic and socioeconomic data.
        demographic_cols (list): List of demographic and socioeconomic column names.
        
    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    # Create a copy to avoid modifying the original
    X = df[demographic_cols].copy()

    print("\n=== FEATURE ENGINEERING ===")

    print("1. Creating per capita features...")
    # Healthcare per capita features
    if all(col in X.columns for col in ['population', 'public_health_facilities']):
        X['health_facilities_per_capita'] = X['public_health_facilities'] / (X['population'] + 0.001)

    if all(col in X.columns for col in ['population', 'public_beds']):
        X['beds_per_capita'] = X['public_beds'] / (X['population'] + 0.001)

    if all(col in X.columns for col in ['population', 'primary_health_centers']):
        X['primary_centers_per_capita'] = X['primary_health_centers'] / (X['population'] + 0.001)

    # Economic per capita features
    if all(col in X.columns for col in ['GDP', 'population']):
        X['GDP_per_capita'] = X['GDP'] / (X['population'] + 0.001)

    print("2. Creating ratio and proportion features...")
    # Urban-Rural ratios
    if all(col in X.columns for col in ['urban_hospitals', 'rural_hospitals']):
        X['urban_rural_hospital_ratio'] = X['urban_hospitals'] / (X['rural_hospitals'] + 0.001)
        X['urbanization_index'] = X['urban_hospitals'] / (X['urban_hospitals'] + X['rural_hospitals'] + 0.001)

    if all(col in X.columns for col in ['urban_beds', 'rural_beds']):
        X['urban_rural_beds_ratio'] = X['urban_beds'] / (X['rural_beds'] + 0.001)
        X['total_beds'] = X['urban_beds'] + X['rural_beds']

    # Healthcare facility ratios
    if all(col in X.columns for col in ['public_beds', 'public_health_facilities']):
        X['beds_per_facility'] = X['public_beds'] / (X['public_health_facilities'] + 0.001)

    # Gender-related features
    if all(col in X.columns for col in ['Male literacy rate %', 'Female literacy rate %']):
        X['literacy_gender_gap'] = X['Male literacy rate %'] - X['Female literacy rate %']
        X['literacy_gender_ratio'] = X['Female literacy rate %'] / (X['Male literacy rate %'] + 0.001)

    print("3. Creating diversity and inequality indices...")
    # Religious diversity (Herfindahl-Hirschman Index)
    religion_cols = ['Hindu', 'Muslim', 'Christian', 'Sikhs', 'Buddhist', 'Others']
    religion_cols = [col for col in religion_cols if col in X.columns]
    
    if len(religion_cols) >= 2:  # Need at least two religion columns for diversity index
        # Normalize to sum to 1
        religion_sum = X[religion_cols].sum(axis=1)
        for col in religion_cols:
            X[f'{col}_norm'] = X[col] / (religion_sum + 0.001)

        # Calculate diversity index (1 - HHI)
        X['religious_diversity'] = 1 - ((X[[f'{col}_norm' for col in religion_cols]]**2).sum(axis=1))

        # Drop the normalized columns as they're not needed anymore
        X = X.drop(columns=[f'{col}_norm' for col in religion_cols])

    # Healthcare diversity
    health_cols = [col for col in X.columns if any(term in col for term in ['health', 'hospital', 'beds'])]
    if len(health_cols) >= 3:  # Only if we have enough healthcare features
        # Create a healthcare infrastructure score
        X_health = X[health_cols].fillna(X[health_cols].median())
        scaler = StandardScaler()
        health_scaled = scaler.fit_transform(X_health)
        X['healthcare_infrastructure_score'] = np.mean(health_scaled, axis=1)

        # Calculate variance in healthcare metrics as a measure of healthcare inequality
        X['healthcare_inequality'] = np.std(health_scaled, axis=1)

    print("4. Applying non-linear transformations...")
    # Log transformations for skewed features
    numeric_cols = X.select_dtypes(include=['number']).columns
    skewed_features = []
    for col in numeric_cols:
        if col in X.columns and X[col].skew() > 1.5:  # Check for highly skewed features
            skewed_features.append(col)

    print(f"   Found {len(skewed_features)} highly skewed features")
    for col in skewed_features:
        min_val = X[col].min()
        # Ensure all values are positive before log transform
        if min_val <= 0:
            X[f'{col}_log'] = np.log(X[col] - min_val + 1)
        else:
            X[f'{col}_log'] = np.log(X[col])

    print("5. Creating interaction features...")
    # Create meaningful interaction terms
    interaction_pairs = [
        # Healthcare and population density
        ('density', 'public_health_facilities'),
        ('density', 'public_beds'),
        # Literacy and healthcare
        ('Average literacy rate %', 'public_health_facilities'),
        # Economic status and healthcare
        ('GDP', 'public_beds'),
        ('GDP', 'public_health_facilities')
    ]

    for col1, col2 in interaction_pairs:
        if col1 in X.columns and col2 in X.columns:
            X[f'{col1}_x_{col2}'] = X[col1] * X[col2]

    print("6. Creating cluster-based features...")
    # Cluster regions based on socioeconomic profiles
    numeric_cols = X.select_dtypes(include=['number']).columns
    if len(numeric_cols) >= 3:  # Need at least a few features for meaningful clustering
        try:
            # Fill missing values for clustering
            X_cluster = X[numeric_cols].fillna(X[numeric_cols].median())

            # Standardize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cluster)

            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            X['socioeconomic_cluster'] = kmeans.fit_predict(X_scaled)

            # Create one-hot encoded cluster features
            cluster_dummies = pd.get_dummies(X['socioeconomic_cluster'], prefix='cluster')
            X = pd.concat([X, cluster_dummies], axis=1)
        except Exception as e:
            print(f"Error in clustering: {str(e)}")

    print("7. Creating PCA-based features...")
    # Apply PCA for dimensionality reduction
    if len(numeric_cols) >= 5:  # Only if we have enough features
        try:
            # Fill missing values for PCA
            X_pca = X[numeric_cols].fillna(X[numeric_cols].median())

            # Standardize data for PCA
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_pca)

            # Apply PCA
            pca = PCA(n_components=0.95)  # Capture 95% of variance
            pca_components = pca.fit_transform(X_scaled)

            # Add PCA components as features
            for i in range(pca_components.shape[1]):
                X[f'pca_component_{i+1}'] = pca_components[:, i]

            print(f"   Added {pca_components.shape[1]} PCA components")
        except Exception as e:
            print(f"Error in PCA: {str(e)}")

    print(f"\nFeature engineering complete! Created {X.shape[1]} features from {len(demographic_cols)} original features.")
    return X

if __name__ == "__main__":
    # If running this script directly, load the merged data and perform feature engineering
    from merge_data import merge_data
    df = merge_data()
    X_engineered, y, existing_demo_socio_cols = engineer_features(df)
    print(f"Engineered features shape: {X_engineered.shape}")