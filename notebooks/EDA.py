"""
Exploratory Data Analysis (EDA) for COVID-19 Dataset.
Includes target distribution, time trends, state-level insights, positivity rate, demographic relationships, and correlations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def eda(df):
    """
    Run exploratory data analysis on the dataset.

    Parameters:
    df (pd.DataFrame): The cleaned and merged COVID-19 dataset.
    """

    print(f"Dataset shape: {df.shape}")
    print("First few rows:")
    print(df.head())

    # ---- Target Variable Distribution ----
    plt.figure(figsize=(10, 4))
    plt.hist(df['cum_positive_cases'].dropna(), bins=50, edgecolor='k')
    plt.title('Cumulative Positive Cases Distribution')
    plt.xlabel('Cumulative Positive Cases')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # ---- Daily Positive Cases Over Time ----
    if 'date' in df.columns and 'daily_positive_cases' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        daily = df.groupby('date')['daily_positive_cases'].sum()

        plt.figure(figsize=(10, 4))
        plt.plot(daily.index, daily.values)
        plt.title('Daily Positive Cases Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cases')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ---- Tests vs Positive Cases Scatter ----
    if 'daily_tests' in df.columns and 'daily_positive_cases' in df.columns:
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x='daily_tests', y='daily_positive_cases', data=df, alpha=0.5)
        plt.title('Daily Tests vs Daily Positive Cases')
        plt.xlabel('Daily Tests')
        plt.ylabel('Daily Positive Cases')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ---- Top States by Cases ----
    if 'state' in df.columns:
        latest = df.sort_values('date').groupby('state').tail(1)
        top_states = latest.sort_values('cum_positive_cases', ascending=False).head(10)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='cum_positive_cases', y='state', data=top_states)
        plt.title('Top 10 States by Cumulative Positive Cases')
        plt.xlabel('Cumulative Cases')
        plt.ylabel('State')
        plt.tight_layout()
        plt.show()

    # ---- Positivity Rate Distribution ----
    if 'cases_per_test' in df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df['cases_per_test'].dropna(), kde=True, bins=40)
        plt.title('Cases per Test (Positivity Rate)')
        plt.xlabel('Cases per Test')
        plt.tight_layout()
        plt.show()

    # ---- GDP per Capita vs Cases ----
    if 'per capita in' in df.columns:
        latest = df.sort_values('date').groupby('state').tail(1)
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x='per capita in', y='cum_positive_cases', data=latest)
        plt.title('GDP per Capita vs Cumulative Positive Cases')
        plt.xlabel('GDP per Capita')
        plt.ylabel('Cumulative Cases')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ---- Correlation Heatmap ----
    numeric_cols = df.select_dtypes(include=['number']).columns
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap (Numerical Features)')
    plt.tight_layout()
    plt.show()
