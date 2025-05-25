import pandas as pd
import os

def clean_data(df = pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by:
    - Ensuring cumulative positive cases are monotonically increasing for Kerala.
    - Correcting negative daily positive cases by replacing them with the mean of the two days before and two days after.
    - Adjusting cumulative positive cases accordingly.
    - Dropping unnecessary columns.
    - Saving the cleaned data to a CSV file.
    All according to insights from the EDA notebook.
    Args:
        pd.DataFrame: The DataFrame to clean.
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Check if the csv file already exists
    cleaned_data_path = "Data/Train/train_data.csv"
    if os.path.exists(cleaned_data_path):
        print(f"Cleaned data already exists at {cleaned_data_path}. Loading it.")
        return pd.read_csv(cleaned_data_path)

    # For Kerala state, go over the rows and make the cum_positive_cases monotonically increasing
    kerala_df = df[df["state"] == "Kerala"].copy()
    kerala_df["cum_positive_cases"] = kerala_df["cum_positive_cases"].cummax()
    df.update(kerala_df)

    # For any state that has a negative value in daily_positive_cases:
    #   - Replace the negative value with the mean of the two days before and two days after (excluding the negative value itself).
    #   - After this correction, adjust cum_positive_cases for that state so it remains consistent with the updated daily values.
    for state in df["state"].unique():
        state_df = df[df["state"] == state].copy()
        negative_indices = state_df[state_df["daily_positive_cases"] < 0].index
        
        for idx in negative_indices:
            # Get the range of indices to consider
            start_idx = max(idx - 2, state_df.index.min())
            end_idx = min(idx + 3, state_df.index.max() + 1)
            mean_value = state_df.loc[start_idx:end_idx, "daily_positive_cases"].mean()
            df.at[idx, "daily_positive_cases"] = mean_value

        # Recalculate cum_positive_cases for the state
        df.loc[df["state"] == state, "cum_positive_cases"] = df.loc[df["state"] == state, "daily_positive_cases"].cumsum()

    # Drop the columns: 
    # cum_positivity_rate, daily_positive_cases, cum_recovered, daily_cases_per_million
    # daily_recovered, cum_deceased, daily_deceased, daily_positivity_rate
    columns_to_drop = [
        "cum_positivity_rate", "daily_positive_cases", "cum_recovered", "daily_cases_per_million",
        "daily_recovered", "cum_deceased", "daily_deceased", "daily_positivity_rate"
    ]
    df.drop(columns=columns_to_drop, inplace=True)

    # Save the cleaned data
    df.to_csv(cleaned_data_path, index=False)

    return df


