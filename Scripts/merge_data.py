"""
Module for merging multiple datasets into a single DataFrame for COVID-19 analysis.
"""

import os
import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np

def merge_data() -> pd.DataFrame:
    """
    Merges multiple cleaned CSV files into a single DataFrame.
    The function performs the following steps:
    1. Check if the merged file already exists:
       - If it exists, load and return the merged file.
       - If it does not exist, process the raw data.
    2. Load and clean the following datasets:
       - allmetrics_states.csv (COVID-19 metrics)
       - state_wise_pop.csv (socio-economic data)
       - HospitalBedsIndia.csv (medicare data)
       - ICMRTestingLabs.csv (testing labs data)
    3. Merge the cleaned DataFrames on the 'state' column using an inner join.
    4. Save the merged DataFrame to the processed folder as a CSV file.
    5. Return the merged DataFrame.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    # Define the path to the output file
    out_path = os.path.join("Data", "Processed", "merged_data.csv")

    # Check if the merged file already exists
    if os.path.exists(out_path):
        print(f"Merged file already exists at {out_path}. Loading it.")
        return pd.read_csv(out_path)

    # Load and clean the allmetrics_states.csv file
    covid_df = load_and_clean_allmetrics() # doesn't have "Lakshadweep"

    # Load and clean the state_wise_pop.csv file
    socio_df = load_and_clean_states_data()

    # Load and clean the medicare data
    medicare_df = load_and_clean_medicare_data() # doesn't have "Dadra and Nagar Haveli and Daman and Diu"

    # Load and clean the ICMRTestingLabs.csv file
    imcr_df = load_and_clean_testing_labs()

    # Define target df
    merged_df = pd.DataFrame()

    # Merge the dataframes on the 'state' column
    merged_df = pd.merge(covid_df, socio_df, on='state', how='inner')
    merged_df = pd.merge(merged_df, medicare_df, on='state', how='inner')
    merged_df = pd.merge(merged_df, imcr_df, on='state', how='inner')

    # Save the merged DataFrame to a CSV file
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged_df.to_csv(out_path, index=False)
    print(f'Merged data saved to {out_path}')

    return merged_df

def load_and_clean_allmetrics() -> pd.DataFrame:
    """
    Load allmetrics_states.csv, clean the data, save it to the processed folder, and return it.
    The function performs the following steps:
    1. Check if the cleaned file already exists:
       - If it exists, load and return the cleaned file.
       - If it does not exist, process the raw data.
    2. Read the raw CSV file.
    3. Add a year to the 'dates' column, starting from 2020, and increment the year when transitioning from December to January.
    4. Convert the 'dates' column to datetime format.
    5. Remove rows where the 'state' column is "India."
    6. Fill any missing values (NaN) with 0.
    7. Filter out rows with dates past 13/8/2021 (including that date).
    8. Keep only the relevant columns.
    9. Save the cleaned DataFrame to the processed folder as a CSV file.
    10. Return the cleaned DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Path to the processed file
    out_path = os.path.join("Data", "Processed", "allmetrics_states_cleaned.csv")

    # Check if the cleaned file already exists
    if os.path.exists(out_path):
        print(f"Cleaned file already exists at {out_path}. Loading it.")
        return pd.read_csv(out_path)

    # The path to the CSV file
    path = os.path.join("Data", "Raw", "Covid_Today", "allmetrics_states.csv")

    # Read the CSV file
    df = pd.read_csv(path)

    # The df has the date in the format "DD Month" without a year
    # Add a year to the dates
    def add_year_to_dates(group: pd.DataFrame) -> pd.DataFrame:
        year = 2020
        previous_month = None
        updated_dates = []

        for date in group['dates']:
            day, month = date.split(' ')
            if previous_month == 'December' and month == 'January':
                year += 1  # Increment the year when transitioning from December to January
            updated_dates.append(f"{day} {month} {year}")
            previous_month = month

        group = group.copy()
        group['dates'] = updated_dates
        return group

    # Apply the function to each state
    df = df.groupby('state').apply(add_year_to_dates)

    # Convert the 'dates' column to datetime format
    df['dates'] = pd.to_datetime(df['dates'], format='%d %B %Y')

    # Remove the state "India"
    df = df[df['state'] != 'India']

    # Fill any NaN values with 0
    df = df.fillna(0)

    # Filter out any dates past 13/8/2021 (including that date)
    cutoff_date = pd.to_datetime('13/8/2021', format='%d/%m/%Y')
    df = df[df['dates'] < cutoff_date]

    # Extract components from datetime for easier analysis
    df['year'] = df['dates'].dt.year
    df['month'] = df['dates'].dt.month
    df['day_of_week'] = df['dates'].dt.dayofweek

    # Keep the columns we want
    df = df[['state', 'dates', 'year', 'month', 'day_of_week', 
             'cum_positive_cases', 'cum_positivity_rate', 'daily_positive_cases',
             'cum_recovered', 'daily_recovered', 'cum_deceased', 'daily_deceased', 'daily_positivity_rate',
             'daily_tests', 'cum_tests', 'test_per_million', 'daily_cases_per_million', 'daily_tests_per_million']]
    
    # Save the cleaned DataFrame to a CSV file
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'Cleaned data saved to {out_path}')

    return df

def load_and_clean_states_data() -> pd.DataFrame:
    """
    Load state_wise_pop.csv, clean the data, save it to the processed folder, and return it.
    The function performs the following steps:
    1. Check if the cleaned file already exists:
       - If it exists, load and return the cleaned file.
       - If it does not exist, process the raw data.
    2. Read the raw CSV file.
    3. Rename columns for consistency and readability.
    4. Remove unnecessary columns.
    5. Replace '&' with 'and' in the 'state' column for uniformity.
    6. Replace any missing values ('-') with NaN.
    7. Impute missing values in numeric columns using KNNImputer.
    8. Add a new column 'density' calculated as 'population' divided by 'area'.
    9. Save the cleaned DataFrame to the processed folder as a CSV file.
    10. Return the cleaned DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Path to the processed file
    out_path = os.path.join("Data", "Processed", "state_wise_pop_cleaned.csv")

    # Check if the cleaned file already exists
    if os.path.exists(out_path):
        print(f"Cleaned file already exists at {out_path}. Loading it.")
        return pd.read_csv(out_path)

    # The path to the CSV file
    path = os.path.join("Data", "Raw", "2024_Indian_States_Data", "state_wise_pop.csv")

    # Read the CSV file
    df = pd.read_csv(path)

    # Rename Column names
    df.rename(columns={
        'States/Uts': 'state',
        'population(2024)': 'population',
        'Male(literacy rate)': 'Male literacy rate %',
        'Female (literacy rate)Average (literacy rate)': 'Female literacy rate %',
        'average (literacy rate)': 'Average literacy rate %',
        'sex ratio (number of female per male)': 'Female to Male ratio',
        'Area (sq. km)': 'area'
        }, inplace=True)

    # Remove columns population(1901), population(1951), population(2011), population(2023)
    df.drop(columns=['population(1901)', 'population(1951)', 'population(2011)', 'population(2023)', 'Majority'], inplace=True)

    # Change any state that has '&' to 'and'
    df['state'] = df['state'].str.replace('&', 'and', regex=False)

    # Replace any missing values with NaN
    df.replace('-', np.nan, inplace=True)

    # Impute using only the numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputer = KNNImputer(n_neighbors=5)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # Add a new column 'density' to the DataFrame
    df['density'] = df['population'] / df['area']

    # Save the cleaned DataFrame to a CSV file
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'Cleaned data saved to {out_path}')

    return df

def load_and_clean_medicare_data() -> pd.DataFrame:
    """
    Load HospitalBedsIndia.csv, clean the data, save it to the processed folder, and return it.
    The function performs the following steps:
    1. Check if the cleaned file already exists:
       - If it exists, load and return the cleaned file.
       - If it does not exist, process the raw data.
    2. Read the raw CSV file.
    3. Drop the 'Sno' column as it is unnecessary.
    4. Rename the 'State/UT' column to 'state' for consistency.
    5. Simplify column names for better readability.
    6. Replace '&' with 'and' in the 'state' column for uniformity.
    7. Replace any missing values ('-') with NaN.
    8. Impute missing values in numeric columns using KNNImputer.
    9. Save the cleaned DataFrame to the processed folder as a CSV file.
    10. Return the cleaned DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Path to the processed file
    out_path = os.path.join("Data", "Processed", "medicare_data_cleaned.csv")

    # Check if the cleaned file already exists
    if os.path.exists(out_path):
        print(f"Cleaned file already exists at {out_path}. Loading it.")
        return pd.read_csv(out_path)
    
    # The path to the CSV file
    path = os.path.join("Data", "Raw", "Covid-19_India_Reports", "COVID-19_in_India", "HospitalBedsIndia.csv")

    # Read the CSV file
    df = pd.read_csv(path)

    # Drop the 'Sno' column
    df.drop(columns=['Sno'], inplace=True)

    # Simplify other column names
    df.rename(columns={
        'State/UT': 'state',
        'NumPrimaryHealthCenters_HMIS': 'primary_health_centers',
        'NumCommunityHealthCenters_HMIS': 'community_health_centers',
        'NumSubDistrictHospitals_HMIS': 'sub_district_hospitals',
        'NumDistrictHospitals_HMIS': 'district_hospitals',
        'TotalPublicHealthFacilities_HMIS': 'public_health_facilities',
        'NumPublicBeds_HMIS': 'public_beds',
        'NumRuralHospitals_NHP18': 'rural_hospitals',
        'NumRuralBeds_NHP18': 'rural_beds',
        'NumUrbanHospitals_NHP18': 'urban_hospitals',
        'NumUrbanBeds_NHP18': 'urban_beds'
    }, inplace=True)

    # Remove any state that has '&' to 'and'
    df['state'] = df['state'].str.replace('&', 'and', regex=False)

    # Replace any missing values with NaN
    df.replace('-', np.nan, inplace=True)

    # Impute using only the numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputer = KNNImputer(n_neighbors=5)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # Save the cleaned DataFrame to a CSV file
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'Cleaned data saved to {out_path}')

    return df

def load_and_clean_testing_labs() -> pd.DataFrame:
    """
    Load ICMRTestingLabs.csv, clean the data, save it to the processed folder, and return it.
    The function performs the following steps:
    1. Check if the cleaned file already exists:
       - If it exists, load and return the cleaned file.
       - If it does not exist, process the raw data.
    2. Read the raw CSV file.
    3. Keep only the 'state' and 'type' columns.
    4. Group by 'state' and count the existing 'type' values.
    5. Replace '&' with 'and' in the 'state' column for uniformity.
    6. Save the cleaned DataFrame to the processed folder as a CSV file.
    7. Return the cleaned DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Path to the processed file
    out_path = os.path.join("Data", "Processed", "ICMRTestingLabs_cleaned.csv")

    # Check if the cleaned file already exists
    if os.path.exists(out_path):
        print(f"Cleaned file already exists at {out_path}. Loading it.")
        return pd.read_csv(out_path)

    # The path to the CSV file
    path = os.path.join("Data", "Raw", "Covid-19_India_Reports", "COVID-19_in_India", "ICMRTestingLabs.csv")

    # Read the CSV file
    df = pd.read_csv(path)

    # Keep only the state column and the type
    df = df[['state', 'type']]

    # Group by state and count the existing types
    df = df.groupby('state')['type'].value_counts().unstack(fill_value=0)

    # Reset the index to ensure 'state' is a regular column
    df = df.reset_index()

    # Replace '&' with 'and' in the 'state' column
    df['state'] = df['state'].str.replace('&', 'and', regex=False)

    # Save the cleaned DataFrame to a CSV file
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'Cleaned data saved to {out_path}')

    return df

if __name__ == "__main__":
    # If running this script directly, perform the merge and return the data
    merged_data = merge_data()
    print(f"Merged data shape: {merged_data.shape}")