import os
import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np

def merge_data(out_path: str = None) -> pd.DataFrame:
    """
    Merges multiple CSV files into a single DataFrame.
    The function reads two CSV files, one containing socio-economic data and the other containing COVID-19 data.
    Args:
        out_path (str): Path to save the merged DataFrame. If None, the DataFrame will not be saved.
    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    # Dynamically determine the project directory based on the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Define paths relative to the project directory
    path_to_socio = os.path.join(project_dir, "Data", "Raw", "2024_Indian_States_Data", "state wise pop.csv")
    path_to_covid = os.path.join(project_dir, "Data", "Raw", "Covid-19_India_Reports", "COVID-19_Corona_Virus_India_Dataset", "complete.csv")

    print(f'Merging data from {path_to_socio} and {path_to_covid}')

    # Read the CSV files
    socio_df = pd.read_csv(path_to_socio)
    covid_df = pd.read_csv(path_to_covid)

    # rename state column name to 'States/Uts'
    covid_df.rename(columns={'Name of State / UT': 'States/Uts'}, inplace=True)

    # Standardize state names
    covid_df['States/Uts'] = covid_df['States/Uts'].str.replace(r'^(Union Territory of )', '', regex=True)
    socio_df['States/Uts'] = socio_df['States/Uts'].str.replace(r'^(Jammu & Kashmir)', 'Jammu and Kashmir', regex=True)
    socio_df['States/Uts'] = socio_df['States/Uts'].str.replace(r'^(Telangana)', 'Telengana', regex=True)

    # Get the columns of the states
    socio_states = socio_df['States/Uts'].unique()
    covid_states = covid_df['States/Uts'].unique()

    # Find the intersection of the two dataframes by state
    intersection_states = set(socio_states).intersection(set(covid_states))

    ### imputation ###
    # Replace non-numeric values (e.g., '-') with NaN
    socio_data = socio_df.drop(columns=['States/Uts','Majority']).replace('-', np.nan)

    print(socio_data.head())

    # Convert all columns to numeric (if possible)
    socio_data = socio_data.apply(pd.to_numeric, errors='coerce')

    print(socio_data.head())

    # Initialize the KNNImputer
    imputer = KNNImputer(n_neighbors=5)

    # Perform imputation
    imputed_data = imputer.fit_transform(socio_data)

    # Convert the imputed data back to a DataFrame
    imputed_socio_df = pd.DataFrame(imputed_data, columns=socio_data.columns)

    # Add the 'States/Uts' column back to the DataFrame
    imputed_socio_df['States/Uts'] = socio_df['States/Uts']
    imputed_socio_df['Majority'] = socio_df['Majority']

    # remove the states that are not in the intersection
    imputed_socio_df = imputed_socio_df[imputed_socio_df['States/Uts'].isin(intersection_states)]
    covid_df = covid_df[covid_df['States/Uts'].isin(intersection_states)]

    # Merge the two DataFrames on 'States/Uts'
    merged_df = pd.merge(imputed_socio_df, covid_df, on='States/Uts', how='inner')

    # Save the merged DataFrame to a CSV file
    if out_path is None:
        out_path = os.path.join(project_dir, "Data", "Processed", "merged_data.csv")

    merged_df.to_csv(out_path, index=False)

    print(f'Merged data saved to {out_path}')

    return merged_df
