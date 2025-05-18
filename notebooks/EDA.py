
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set the file path - update this to your file location
file_path = r'Data\Processed\merged_data.csv'  # Change to your path

# Load the data
df = pd.read_csv(file_path)
print(f"Dataset loaded with shape: {df.shape}")

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
demo_data = latest_data[latest_data['population'] > 0].copy()

# Descriptive statistics for demographic variables
print("\nDescriptive Statistics for Demographic Variables:")
print(demo_data[demographic_socioeconomic_cols].describe().T)

# Analyze relationship between population density and COVID-19 metrics
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

# Healthcare facilities per million
for col in healthcare_cols:
    if col in demo_data.columns:
        demo_data[f'{col}_per_million'] = demo_data[col] / demo_data['population'] * 1000000

# Analyze top 10 states by healthcare infrastructure
healthcare_per_million_cols = [col + '_per_million' for col in healthcare_cols if col in demo_data.columns]

# Bar chart for healthcare infrastructure per million for top 10 states by cases
plt.figure(figsize=(16, 10))
top_10_states = demo_data.sort_values('cases_per_million', ascending=False).head(10)['state'].values

# Melt the dataframe for easier plotting
healthcare_melted = pd.melt(
    demo_data[demo_data['state'].isin(top_10_states)][['state'] + healthcare_per_million_cols[:4]],
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
healthcare_correlation = demo_data[healthcare_per_million_cols + ['cases_per_million', 'deaths_per_million']].corr()

plt.figure(figsize=(14, 12))
sns.heatmap(healthcare_correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation: Healthcare Infrastructure and COVID-19 Metrics')
plt.tight_layout()
plt.show()

# 6. Religious Demographics Analysis
print("\n----- Religious Demographics Analysis -----")

religion_cols = ['Hindu', 'Muslim', 'Christian', 'Sikhs', 'Buddhist', 'Others']

# Check if we have religious demographic data
if all(col in demo_data.columns for col in religion_cols):
    # Scatter plots for religious demographics vs COVID-19 metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, religion in enumerate(religion_cols):
        axes[i].scatter(demo_data[religion], demo_data['cases_per_million'])
        axes[i].set_title(f'{religion} Population % vs Cases per Million')
        axes[i].set_xlabel(f'{religion} Population %')
        axes[i].set_ylabel('Cases per Million')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Correlation between religious demographics and COVID-19 metrics
    religion_correlation = demo_data[religion_cols + ['cases_per_million', 'deaths_per_million', 'tests_per_million']].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(religion_correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation: Religious Demographics and COVID-19 Metrics')
    plt.tight_layout()
    plt.show()

# Create a copy of the dataset for modeling
df_model = df.copy()

# Get only numeric columns
numeric_cols = df_model.select_dtypes(include=['int64', 'float64']).columns.tolist()
df_model = df_model[numeric_cols]

print(f"Using {len(numeric_cols)} numeric columns for modeling")
print(f"Shape of data for modeling: {df_model.shape}")
# Check for any remaining missing values in the modeling dataset
missing_values = df_model.isnull().sum()
if missing_values.sum() > 0:
    print("\nWarning: Missing values in numeric columns:")
    print(missing_values[missing_values > 0])
    print("\nFilling missing values with median")
    df_model = df_model.fillna(df_model.median())
# 7. Correlation Analysis
print("\n----- Correlation Analysis -----")

# Calculate correlation matrix
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
corr_with_target = correlation_matrix['cum_positive_cases'].sort_values(ascending=False)
print("\nTop correlations with cum_positive_cases:")
print(corr_with_target[:11])  # Top 10 + itself

# Plot correlations of demographic and socioeconomic features with target
plt.figure(figsize=(14, 10))
demo_socio_corr = correlation_matrix.loc[demographic_socioeconomic_cols, ['cum_positive_cases']]
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
monthly_pattern = df.groupby('month')[['daily_positive_cases']].mean().reset_index()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_pattern['month_name'] = monthly_pattern['month'].apply(lambda x: months[x-1])

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
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['daily_positivity_rate'].dropna(), kde=True)
plt.title('Daily Positivity Rate Distribution')
plt.xlabel('Daily Positivity Rate')

plt.subplot(1, 2, 2)
plt.plot(time_series['dates'], time_series['daily_positive_cases'] / time_series['daily_tests'])
plt.title('Daily Positivity Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Positivity Rate')
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()

# 10. Test Accessibility Analysis
print("\n----- Test Accessibility Analysis -----")

# Tests per million vs GDP per capita
plt.figure(figsize=(10, 6))
plt.scatter(demo_data['per capita in'], demo_data['tests_per_million'])
plt.title('GDP per Capita vs Tests per Million')
plt.xlabel('GDP per Capita')
plt.ylabel('Tests per Million')
plt.grid(True)
plt.tight_layout()
plt.show()

# Function to plot time series for a few selected states
def plot_time_series(dataframe, states, feature):
    plt.figure(figsize=(14, 8))
    for state in states:
        state_data = dataframe[dataframe['state'] == state]
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
top_states = df.groupby('state')['cum_positive_cases'].max().sort_values(ascending=False).head(5).index
plot_time_series(df, top_states, 'cum_positive_cases')
plot_time_series(df, top_states, 'daily_positive_cases')
plot_time_series(df, top_states, 'daily_tests_per_million')