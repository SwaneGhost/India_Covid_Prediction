import pandas as pd
import numpy as np


def enhanced_data():
    """
    Loads and enhances COVID-19 data with new time-based and statistical features.

    This function adds lag features, rolling averages, growth rates,
    and categorical binning for improved machine learning performance.

    It also cleans and clips the data to avoid errors during training.

    Returns:
        DataFrame: The enhanced dataset ready for model input.
    """

    # Load the updated dataset (pre-merged)
    df = pd.read_csv('../Data/Processed/update.csv')

    # Convert date strings to datetime objects
    df['date'] = pd.to_datetime(df['dates'])

    # Sort rows by state and date to ensure proper time sequence
    df = df.sort_values(by=['state', 'date'])

    # Group by state to apply time-based transformations per group
    group = df.groupby('state')

    # Time features
    df['dayofweek'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
    df['month'] = df['date'].dt.month  # 1 to 12

    # Lag feature
    df['daily_tests_lag1'] = group['daily_tests'].shift(1)  # previous day's tests

    # Rolling average
    df['rolling_tests_7d'] = group['daily_tests'].transform(lambda x: x.rolling(7).mean())  # 7-day average

    # Growth rate
    df['test_growth_rate'] = (df['daily_tests'] - df['daily_tests_lag1']) / (df['daily_tests_lag1'] + 1e-5)

    # Cases per test
    df['cases_per_test'] = df['target'] / (df['daily_tests'] + 1e-5)

    #  Time since first reported case
    df['days_since_first_case'] = group['target'].transform(lambda x: x.ne(0).cumsum())

    #  Positivity level (categorical bin)
    df['positivity_level'] = pd.qcut(df['cases_per_test'], 4, labels=False, duplicates='drop')

    # Drop rows that have NaNs (usually caused by lag/rolling)
    df = df.dropna().reset_index(drop=True)

    # Clip outliers to prevent extreme values
    df['test_growth_rate'] = df['test_growth_rate'].clip(-10, 10)
    df['cases_per_test'] = df['cases_per_test'].clip(0, 1)

    # Summary printout
    print(f"Enhanced Denis dataset shape: {df.shape}")
    print("New columns added:", [
        'dayofweek', 'month',
        'daily_tests_lag1', 'rolling_tests_7d',
        'test_growth_rate', 'cases_per_test',
        'days_since_first_case', 'positivity_level'
    ])

    # Save enhanced version for future use
    df.to_csv("../Data/Processed/enhanced_covid_data2.csv", index=False)

    return df
