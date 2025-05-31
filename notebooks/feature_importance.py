"""
Lightweight Feature Importance Analysis for COVID-19 Model.
Includes:
- Coefficient-based importance
- Permutation-based importance
- Performance metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


def run_feature_importance(model, df, target_col='cum_positive_cases', test_size=0.2):
    """
    Runs feature importance and performance analysis.

    Includes:
    - Coefficient-based importance for linear models
    - Permutation importance for general models
    - R2 and RMSE metrics for training and testing

    Parameters:
        model (Pipeline): Trained pipeline model
        df (DataFrame): Input dataset including features and target
        target_col (str): Name of the target column
        test_size (float): Fraction of data to use as test set
    """

    print("Running Lightweight Feature Importance Analysis")
    print("=" * 60)

    # Prepare data
    id_cols = ['dates', 'date']
    X = df.drop(columns=[target_col] + [col for col in id_cols if col in df.columns])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Coefficient-based importance
    if hasattr(model.named_steps['regressor'], 'coef_'):
        reg = model.named_steps['regressor']
        coefs = reg.coef_

        try:
            feature_names = model.named_steps['preprocessor'].get_feature_names_out().tolist()
        except:
            feature_names = [f'feature_{i}' for i in range(len(coefs))]

        if len(coefs) != len(feature_names):
            print("Mismatch between coefficients and feature names")
            return

        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefs,
            'abs_coefficient': np.abs(coefs)
        }).sort_values('abs_coefficient', ascending=False)

        print("Top Coefficients:")
        print(coef_df[['feature', 'coefficient']].head(10))

        plt.figure(figsize=(10, 6))
        sns.barplot(x='coefficient', y='feature', data=coef_df.head(15))
        plt.title("Top lasso Coefficients")
        plt.axvline(0, color='red', linestyle='--')
        plt.tight_layout()
        plt.show()

    # Permutation importance
    print("Calculating Permutation Importance on Test Set")
    try:
        perm = permutation_importance(
            model, X_test, y_test,
            scoring='r2',
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )

        perm_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': perm.importances_mean,
            'importance_std': perm.importances_std
        }).sort_values('importance_mean', ascending=False)

        print("Top Permutation Importances:")
        print(perm_df[['feature', 'importance_mean']].head(10))

        plt.figure(figsize=(10, 6))
        sns.barplot(
            x='importance_mean',
            y='feature',
            data=perm_df.head(15),
            xerr=perm_df.head(15)['importance_std']
        )
        plt.title("Permutation Importance (Test Set)")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error in permutation importance: {e}")

    # Performance summary
    try:
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
        test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)

        print("Performance Metrics")
        print(f"Train R2: {train_r2:.4f}")
        print(f"Test R2: {test_r2:.4f}")
        print(f"Train RMSE: {train_rmse:.2f}")
        print(f"Test RMSE: {test_rmse:.2f}")
        print(f"Overfitting gap (R2): {train_r2 - test_r2:.4f}")

    except Exception as e:
        print(f"Error calculating performance metrics: {e}")
