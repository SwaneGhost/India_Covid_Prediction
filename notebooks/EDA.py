"""
Module for conducting Exploratory Data Analysis on the COVID-19 dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(df):
    """
    Perform exploratory data analysis on the COVID-19 dataset.
    
    Args:
        df (pd.DataFrame): The merged dataset containing COVID-19 data.
    """
    print(f"Dataset loaded with shape: {df.shape}")

    # Define demographic and socioeconomic columns
    demographic_socioeconomic_cols = [
        'population', 'Male literacy rate %', 'Female literacy rate %',
        'Average literacy rate %', 'GDP', 'per capita in', 'Female to Male ratio',
        'Hindu', 'Muslim', 'Christian', 'Sikhs', 'Buddhist', 'Others', 'density',
        'primary_health_centers', 'community_health_centers', 'sub_district_hospitals',
        'district_hospitals', 'public_health_facilities', 'public_beds',
        'rural_hospitals', 'rural_beds', 'urban_hospitals', 'urban_beds',
        'state'
    ]

    # Check which demographic and socioeconomic columns exist in the dataset
    existing_demo_socio_cols = [col for col in demographic_socioeconomic_cols if col in df.columns]
    print(f"\nExisting demographic and socioeconomic columns: {len(existing_demo_socio_cols)}")
    print(existing_demo_socio_cols)

    # Display basic information about the dataset
    print("\nDataset Overview:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())

    # 1. Target Variable Analysis
    print("\n----- Target Variable Analysis -----")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Cumulative Positive Cases Distribution')
    sns.histplot(df['cum_positive_cases'], kde=True)

    plt.subplot(1, 2, 2)
    plt.title('Cumulative Positive Cases Distribution (Log Scale)')
    sns.histplot(np.log1p(df['cum_positive_cases']), kde=True)
    plt.xlabel('Log(Cumulative Positive Cases + 1)')
    plt.tight_layout()
    plt.show()

    # Print basic statistics about the target variable
    print("\nTarget Variable Statistics:")
    print(df['cum_positive_cases'].describe())
    print(f"Skewness: {df['cum_positive_cases'].skew()}")
    print(f"Kurtosis: {df['cum_positive_cases'].kurt()}")

    # 2. Time Series Analysis
    print("\n----- Time Series Analysis -----")

    # Ensure dates column is in datetime format
    if df['dates'].dtype != 'datetime64[ns]':
        df['dates'] = pd.to_datetime(df['dates'])

    # Aggregate data by date
    time_series = df.groupby('dates')[['cum_positive_cases', 'daily_positive_cases', 
                                      'cum_deceased', 'daily_deceased', 
                                      'cum_tests', 'daily_tests']].sum().reset_index()

    # Plot time series for key metrics
    plt.figure(figsize=(16, 20))

    metrics = ['cum_positive_cases', 'daily_positive_cases', 'cum_deceased', 
               'daily_deceased', 'cum_tests', 'daily_tests']
    titles = ['Cumulative Positive Cases', 'Daily Positive Cases', 'Cumulative Deceased', 
              'Daily Deceased', 'Cumulative Tests', 'Daily Tests']

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(3, 2, i+1)
        plt.plot(time_series['dates'], time_series[metric])
        plt.title(title)
        plt.xticks(rotation=45)
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Monthly trend analysis
    if all(col in df.columns for col in ['year', 'month']):
        monthly_data = df.groupby(['year', 'month'])[['daily_positive_cases', 'daily_deceased', 'daily_tests']].mean().reset_index()
        monthly_data['yearmonth'] = monthly_data['year'].astype(str) + '-' + monthly_data['month'].astype(str).str.zfill(2)

        plt.figure(figsize=(14, 6))
        plt.plot(monthly_data['yearmonth'], monthly_data['daily_positive_cases'], marker='o')
        plt.title('Average Daily Positive Cases by Month')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 3. State-wise Analysis
    print("\n----- State-wise Analysis -----")

    # Get latest data for each state
    latest_date = df['dates'].max()
    latest_data = df[df['dates'] == latest_date]

    # Top 10 states by cumulative positive cases
    top_states_by_cases = latest_data.sort_values('cum_positive_cases', ascending=False).head(10)

    plt.figure(figsize=(14, 8))
    sns.barplot(x='cum_positive_cases', y='state', data=top_states_by_cases)
    plt.title('Top 10 States by Cumulative Positive Cases')
    plt.tight_layout()
    plt.show()

    # Calculate per capita metrics for states
    if 'population' in latest_data.columns:
        latest_data['cases_per_million'] = latest_data['cum_positive_cases'] / latest_data['population'] * 1000000
        latest_data['tests_per_million'] = latest_data['cum_tests'] / latest_data['population'] * 1000000
        latest_data['deaths_per_million'] = latest_data['cum_deceased'] / latest_data['population'] * 1000000

        # Top 10 states by cases per million
        top_states_per_capita = latest_data.sort_values('cases_per_million', ascending=False).head(10)

        plt.figure(figsize=(14, 8))
        sns.barplot(x='cases_per_million', y='state', data=top_states_per_capita)
        plt.title('Top 10 States by COVID-19 Cases per Million')
        plt.tight_layout()
        plt.show()

    # 4. Demographic & Socioeconomic Analysis
    print("\n----- Demographic & Socioeconomic Analysis -----")

    # Filter out rows with zero population to avoid division by zero
    if 'population' in latest_data.columns:
        demo_data = latest_data[latest_data['population'] > 0].copy()

        # Descriptive statistics for demographic variables
        print("\nDescriptive Statistics for Demographic Variables:")
        demo_cols = [col for col in existing_demo_socio_cols if col in demo_data.columns]
        print(demo_data[demo_cols].describe().T)

        # Analyze relationship between population density and COVID-19 metrics
        if 'density' in demo_data.columns and 'cases_per_million' in demo_data.columns:
            plt.figure(figsize=(18, 6))

            plt.subplot(1, 3, 1)
            plt.scatter(demo_data['density'], demo_data['cases_per_million'])
            plt.title('Population Density vs Cases per Million')
            plt.xlabel('Population Density')
            plt.ylabel('Cases per Million')
            plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.scatter(demo_data['density'], demo_data['tests_per_million'])
            plt.title('Population Density vs Tests per Million')
            plt.xlabel('Population Density')
            plt.ylabel('Tests per Million')
            plt.grid(True)

            plt.subplot(1, 3, 3)
            plt.scatter(demo_data['density'], demo_data['deaths_per_million'])
            plt.title('Population Density vs Deaths per Million')
            plt.xlabel('Population Density')
            plt.ylabel('Deaths per Million')
            plt.grid(True)

            plt.tight_layout()
            plt.show()

        # Analyze relationship between literacy rate and COVID-19 metrics
        if 'Average literacy rate %' in demo_data.columns and 'cases_per_million' in demo_data.columns:
            plt.figure(figsize=(18, 6))

            plt.subplot(1, 3, 1)
            plt.scatter(demo_data['Average literacy rate %'], demo_data['cases_per_million'])
            plt.title('Literacy Rate vs Cases per Million')
            plt.xlabel('Average Literacy Rate (%)')
            plt.ylabel('Cases per Million')
            plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.scatter(demo_data['Average literacy rate %'], demo_data['tests_per_million'])
            plt.title('Literacy Rate vs Tests per Million')
            plt.xlabel('Average Literacy Rate (%)')
            plt.ylabel('Tests per Million')
            plt.grid(True)

            plt.subplot(1, 3, 3)
            plt.scatter(demo_data['Average literacy rate %'], demo_data['deaths_per_million'])
            plt.title('Literacy Rate vs Deaths per Million')
            plt.xlabel('Average Literacy Rate (%)')
            plt.ylabel('Deaths per Million')
            plt.grid(True)

            plt.tight_layout()
            plt.show()

        # Analyze relationship between GDP per capita and COVID-19 metrics
        if 'per capita in' in demo_data.columns and 'cases_per_million' in demo_data.columns:
            plt.figure(figsize=(18, 6))

            plt.subplot(1, 3, 1)
            plt.scatter(demo_data['per capita in'], demo_data['cases_per_million'])
            plt.title('GDP per Capita vs Cases per Million')
            plt.xlabel('GDP per Capita')
            plt.ylabel('Cases per Million')
            plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.scatter(demo_data['per capita in'], demo_data['tests_per_million'])
            plt.title('GDP per Capita vs Tests per Million')
            plt.xlabel('GDP per Capita')
            plt.ylabel('Tests per Million')
            plt.grid(True)

            plt.subplot(1, 3, 3)
            plt.scatter(demo_data['per capita in'], demo_data['deaths_per_million'])
            plt.title('GDP per Capita vs Deaths per Million')
            plt.xlabel('GDP per Capita')
            plt.ylabel('Deaths per Million')
            plt.grid(True)

            plt.tight_layout()
            plt.show()

    # 5. Healthcare Infrastructure Analysis
    print("\n----- Healthcare Infrastructure Analysis -----")

    healthcare_cols = ['primary_health_centers', 'community_health_centers', 
                       'sub_district_hospitals', 'district_hospitals', 
                       'public_health_facilities', 'public_beds', 
                       'rural_hospitals', 'rural_beds', 
                       'urban_hospitals', 'urban_beds']

    # Keep only healthcare columns that exist in the data
    healthcare_cols = [col for col in healthcare_cols if col in df.columns]

    if healthcare_cols and 'population' in demo_data.columns:
        # Healthcare facilities per million
        for col in healthcare_cols:
            if col in demo_data.columns:
                demo_data[f'{col}_per_million'] = demo_data[col] / demo_data['population'] * 1000000

        # Healthcare per million columns
        healthcare_per_million_cols = [col + '_per_million' for col in healthcare_cols if col in demo_data.columns]

        if healthcare_per_million_cols:
            # Bar chart for healthcare infrastructure per million for top 10 states by cases
            plt.figure(figsize=(16, 10))
            top_10_states = demo_data.sort_values('cases_per_million', ascending=False).head(10)['state'].values

            # Limit to first 4 healthcare metrics to keep chart readable
            display_cols = healthcare_per_million_cols[:4] if len(healthcare_per_million_cols) >= 4 else healthcare_per_million_cols

            # Melt the dataframe for easier plotting
            healthcare_melted = pd.melt(
                demo_data[demo_data['state'].isin(top_10_states)][['state'] + display_cols],
                id_vars=['state'],
                var_name='Healthcare Facility',
                value_name='Per Million'
            )

            sns.barplot(x='state', y='Per Million', hue='Healthcare Facility', data=healthcare_melted)
            plt.title('Healthcare Facilities per Million in Top 10 States by COVID-19 Cases')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            # Correlation between healthcare infrastructure and COVID-19 metrics
            correlation_cols = healthcare_per_million_cols + ['cases_per_million', 'deaths_per_million']
            correlation_cols = [col for col in correlation_cols if col in demo_data.columns]
            
            if len(correlation_cols) > 1:  # Need at least 2 columns for correlation
                healthcare_correlation = demo_data[correlation_cols].corr()

                plt.figure(figsize=(14, 12))
                sns.heatmap(healthcare_correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
                plt.title('Correlation: Healthcare Infrastructure and COVID-19 Metrics')
                plt.tight_layout()
                plt.show()

    # 6. Religious Demographics Analysis
    print("\n----- Religious Demographics Analysis -----")

    religion_cols = ['Hindu', 'Muslim', 'Christian', 'Sikhs', 'Buddhist', 'Others']
    religion_cols = [col for col in religion_cols if col in df.columns]

    # Check if we have religious demographic data
    if religion_cols and len(religion_cols) >= 2 and 'cases_per_million' in demo_data.columns:
        # Scatter plots for religious demographics vs COVID-19 metrics
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, religion in enumerate(religion_cols):
            if i < len(axes):  # Make sure we don't exceed the number of subplot axes
                axes[i].scatter(demo_data[religion], demo_data['cases_per_million'])
                axes[i].set_title(f'{religion} Population % vs Cases per Million')
                axes[i].set_xlabel(f'{religion} Population %')
                axes[i].set_ylabel('Cases per Million')
                axes[i].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Correlation between religious demographics and COVID-19 metrics
        corr_cols = religion_cols + ['cases_per_million', 'deaths_per_million', 'tests_per_million']
        corr_cols = [col for col in corr_cols if col in demo_data.columns]
        
        if len(corr_cols) > 1:  # Need at least 2 columns for correlation
            religion_correlation = demo_data[corr_cols].corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(religion_correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title('Correlation: Religious Demographics and COVID-19 Metrics')
            plt.tight_layout()
            plt.show()

    # 7. Correlation Analysis
    print("\n----- Correlation Analysis -----")

    # Identify numeric columns for correlation analysis
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"Using {len(numeric_cols)} numeric columns for correlation analysis")

    # Calculate correlation matrix using only numeric columns
    correlation_matrix = df[numeric_cols].corr()

    # Plot heatmap for correlation matrix
    plt.figure(figsize=(18, 16))
    mask = np.triu(correlation_matrix)
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

    # Top 10 features correlated with cum_positive_cases
    if 'cum_positive_cases' in correlation_matrix.columns:
        corr_with_target = correlation_matrix['cum_positive_cases'].sort_values(ascending=False)
        print("\nTop correlations with cum_positive_cases:")
        print(corr_with_target[:11])  # Top 10 + itself

        # Plot correlations of demographic and socioeconomic features with target
        demo_socio_cols = [col for col in existing_demo_socio_cols if col in correlation_matrix.index and col != 'cum_positive_cases']
        
        if demo_socio_cols:
            plt.figure(figsize=(14, 10))
            demo_socio_corr = correlation_matrix.loc[demo_socio_cols, ['cum_positive_cases']]
            demo_socio_corr = demo_socio_corr.sort_values('cum_positive_cases', ascending=False)
            sns.barplot(x=demo_socio_corr['cum_positive_cases'], y=demo_socio_corr.index)
            plt.title('Correlation of Demographic & Socioeconomic Features with Cumulative Positive Cases')
            plt.xlabel('Correlation Coefficient')
            plt.axvline(x=0, color='r', linestyle='--')
            plt.tight_layout()
            plt.show()

    # 8. Temporal Patterns
    print("\n----- Temporal Patterns Analysis -----")

    # Daily patterns
    if 'day_of_week' in df.columns:
        daily_pattern = df.groupby('day_of_week')[['daily_positive_cases']].mean().reset_index()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_pattern['day_name'] = daily_pattern['day_of_week'].apply(lambda x: days[x])

        plt.figure(figsize=(10, 6))
        sns.barplot(x='day_name', y='daily_positive_cases', data=daily_pattern)
        plt.title('Average Daily Positive Cases by Day of Week')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Monthly patterns
    if 'month' in df.columns:
        monthly_pattern = df.groupby('month')[['daily_positive_cases']].mean().reset_index()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_pattern['month_name'] = monthly_pattern['month'].apply(lambda x: months[x-1] if 1 <= x <= 12 else 'Unknown')

        plt.figure(figsize=(10, 6))
        sns.barplot(x='month_name', y='daily_positive_cases', data=monthly_pattern)
        plt.title('Average Daily Positive Cases by Month')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # 9. Testing vs Cases Analysis
    print("\n----- Testing vs Cases Analysis -----")

    # Relationship between testing and positive cases
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(df['daily_tests'], df['daily_positive_cases'], alpha=0.5)
    plt.title('Daily Tests vs Daily Positive Cases')
    plt.xlabel('Daily Tests')
    plt.ylabel('Daily Positive Cases')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(df['cum_tests'], df['cum_positive_cases'], alpha=0.5)
    plt.title('Cumulative Tests vs Cumulative Positive Cases')
    plt.xlabel('Cumulative Tests')
    plt.ylabel('Cumulative Positive Cases')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Positivity rate analysis
    if 'daily_positivity_rate' in df.columns:
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        sns.histplot(df['daily_positivity_rate'].dropna(), kde=True)
        plt.title('Daily Positivity Rate Distribution')
        plt.xlabel('Daily Positivity Rate')

        # Calculate positivity rate for time series
        if 'daily_tests' in time_series.columns and 'daily_positive_cases' in time_series.columns:
            positivity_series = time_series['daily_positive_cases'] / time_series['daily_tests']
            
            plt.subplot(1, 2, 2)
            plt.plot(time_series['dates'], positivity_series)
            plt.title('Daily Positivity Rate Over Time')
            plt.xlabel('Date')
            plt.ylabel('Positivity Rate')
            plt.xticks(rotation=45)
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    # Function to plot time series for a few selected states
    def plot_time_series(dataframe, states, feature):
        if not states or feature not in dataframe.columns:
            return
            
        plt.figure(figsize=(14, 8))
        for state in states:
            state_data = dataframe[dataframe['state'] == state]
            if not state_data.empty:
                plt.plot(state_data['dates'], state_data[feature], label=state)
        plt.title(f'{feature} Over Time for Selected States')
        plt.xlabel('Date')
        plt.ylabel(feature)
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Get top 5 states by cumulative positive cases
    top_states = df.groupby('state')['cum_positive_cases'].max().sort_values(ascending=False).head(5).index.tolist()
    plot_time_series(df, top_states, 'cum_positive_cases')
    plot_time_series(df, top_states, 'daily_positive_cases')
    
    if 'daily_tests_per_million' in df.columns:
        plot_time_series(df, top_states, 'daily_tests_per_million')

    # Return the demographic and socioeconomic columns for use in feature engineering
    return existing_demo_socio_cols

if __name__ == "__main__":
    # If running this script directly, load the merged data and perform EDA
    from merge_data import merge_data
    df = merge_data()
    perform_eda(df)