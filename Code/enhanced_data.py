import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#
# def enhanced_data():
#     # Load original merged dataset
#     df = pd.read_csv('../Data/Processed/train_data.csv')
#
#     # Ensure 'dates' is datetime and sort properly
#     df['date'] = pd.to_datetime(df['dates'])
#     df = df.sort_values(by=['state', 'date'])
#
#     # ========== TIME FEATURES ==========
#     df['dayofweek'] = df['date'].dt.dayofweek
#     df['month'] = df['date'].dt.month
#
#     # ========== LAG & ROLLING ==========
#     group = df.groupby('state')
#     df['daily_positive_cases_lag1'] = group['daily_positive_cases'].shift(1)
#     df['daily_tests_lag1'] = group['daily_tests'].shift(1)
#     df['rolling_cases_7d'] = group['daily_positive_cases'].transform(lambda x: x.rolling(7).mean())
#     df['rolling_tests_7d'] = group['daily_tests'].transform(lambda x: x.rolling(7).mean())
#
#     # ========== GROWTH FEATURES ==========
#     df['case_growth_rate'] = (df['daily_positive_cases'] - df['daily_positive_cases_lag1']) / (
#                 df['daily_positive_cases_lag1'] + 1e-5)
#     df['test_growth_rate'] = (df['daily_tests'] - df['daily_tests_lag1']) / (df['daily_tests_lag1'] + 1e-5)
#
#     # ========== CASES PER TEST ==========
#     df['cases_per_test'] = df['daily_positive_cases'] / (df['daily_tests'] + 1e-5)
#
#     # ========== DAYS SINCE FIRST CASE ==========
#     df['days_since_first_case'] = group['cum_positive_cases'].transform(lambda x: x.ne(0).cumsum())
#
#     # ========== POSITIVITY RATE BINNING ==========
#     df['positivity_level'] = pd.qcut(df['cum_positivity_rate'], 4, labels=False)
#
#     # ========== DROP NAs ==========
#     df = df.dropna().reset_index(drop=True)
#
#     # ========== FIX STRANGE VALUES ==========
#     # Clip negative daily values
#     for col in ['daily_positive_cases', 'daily_recovered', 'daily_deceased',
#                 'rolling_cases_7d', 'rolling_tests_7d']:
#         df[col] = df[col].clip(lower=0)
#
#     # Clip growth rates and positivity ratio
#     df['case_growth_rate'] = df['case_growth_rate'].clip(-10, 10)
#     df['test_growth_rate'] = df['test_growth_rate'].clip(-10, 10)
#     df['cases_per_test'] = df['cases_per_test'].clip(0, 1)
#
#     # ====== Define Features & Target ======
#     target = 'cum_positive_cases'
#     X = df.drop(columns=[target, 'date', 'dates'])
#     y = df[target]
#
#     # Identify categorical and numerical features
#     categorical_features = ['state']
#     numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.difference(categorical_features).tolist()
#
#     # ====== Preprocessing Pipeline ======
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', Pipeline([
#                 ('imputer', SimpleImputer(strategy='mean')),
#                 ('scaler', StandardScaler())
#             ]), numerical_features),
#             ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#         ]
#     )
#
#     # Save enhanced version
#     df.to_csv("../Data/Processed/enhanced_covid_data2.csv", index=False)
#
#     # Print overview
#     print(f"Enhanced dataset shape: {df.shape}")
#     print("New columns added:", [
#         'dayofweek', 'month',
#         'daily_positive_cases_lag1', 'daily_tests_lag1',
#         'rolling_cases_7d', 'rolling_tests_7d',
#         'case_growth_rate', 'test_growth_rate',
#         'cases_per_test', 'days_since_first_case', 'positivity_level'
#     ])
#
#     return df


###############################################################################################################


import pandas as pd
import numpy as np

def enhanced_data():
    # Load the Denis dataset
    df = pd.read_csv('../Data/Processed/update.csv')

    # Ensure date format and sort
    df['date'] = pd.to_datetime(df['dates'])
    df = df.sort_values(by=['state', 'date'])

    # Group by state for time series operations
    group = df.groupby('state')

    # Add time-based features
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    # Lag features
    df['daily_tests_lag1'] = group['daily_tests'].shift(1)

    # Rolling averages
    df['rolling_tests_7d'] = group['daily_tests'].transform(lambda x: x.rolling(7).mean())

    # Growth features
    df['test_growth_rate'] = (df['daily_tests'] - df['daily_tests_lag1']) / (df['daily_tests_lag1'] + 1e-5)

    # Cases per test
    df['cases_per_test'] = df['target'] / (df['daily_tests'] + 1e-5)

    # Days since first target > 0
    df['days_since_first_case'] = group['target'].transform(lambda x: x.ne(0).cumsum())

    # Positivity level binning (on cases per test, as cum_positivity_rate is missing)
    df['positivity_level'] = pd.qcut(df['cases_per_test'], 4, labels=False, duplicates='drop')

    # Drop rows with any missing values (after lagging)
    df = df.dropna().reset_index(drop=True)

    # Clip outliers
    df['test_growth_rate'] = df['test_growth_rate'].clip(-10, 10)
    df['cases_per_test'] = df['cases_per_test'].clip(0, 1)

    print(f"Enhanced Denis dataset shape: {df.shape}")
    print("New columns added:", [
        'dayofweek', 'month',
        'daily_tests_lag1', 'rolling_tests_7d',
        'test_growth_rate', 'cases_per_test',
        'days_since_first_case', 'positivity_level'
    ])

    # Save version if needed
    df.to_csv("../Data/Processed/enhanced_covid_data2.csv", index=False)

    return df