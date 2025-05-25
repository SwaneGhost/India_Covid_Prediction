# # === enhance_lasso_strategies.py ===
# """
# Enhanced Lasso Regression Module
# Focuses on Strategy 2 (Feature Selection) and Strategy 3 (Target Transformation)
# Corrected: Feature selector now applied post-preprocessing to avoid string errors.
# """
#
# import numpy as np
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
# import shap
# from sklearn.ensemble import VotingRegressor
# from sklearn.linear_model import LassoCV
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, OrdinalEncoder
# from sklearn.impute import SimpleImputer, KNNImputer
# from sklearn.feature_selection import SelectKBest, f_regression
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from scipy import stats
# from sklearn.base import clone
# import warnings
# warnings.filterwarnings('ignore')
#
#
# def enhance_lasso_strategies(X_train, X_test, y_train, y_test, categorical_features, numerical_features, ensemble=False):
#     results = {}
#     models = {}
#
#     print("\n1. Strategy: Feature Selection")
#     model2, metrics2 = strategy_feature_selection(X_train, X_test, y_train, y_test,
#                                                   categorical_features, numerical_features)
#     results['Strategy 2: Feature Selection'] = metrics2
#     models['Strategy 2'] = model2
#
#     print("\n2. Strategy: Target Transformation")
#     model3, metrics3 = strategy_target_transformation(X_train, X_test, y_train, y_test,
#                                                       categorical_features, numerical_features)
#     results['Strategy 3: Target Transformation'] = metrics3
#     models['Strategy 3'] = model3
#
#     if ensemble:
#         print("\n3. Strategy: Ensemble of Strategies 2 & 3")
#         ensemble_model = VotingRegressor(estimators=[
#             ('strat2', model2),
#             ('strat3', model3)
#         ])
#         ensemble_model.fit(X_train, y_train)
#         y_pred = ensemble_model.predict(X_test)
#         metrics_ensemble = {
#             'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
#             'r2': r2_score(y_test, y_pred),
#             'mae': mean_absolute_error(y_test, y_pred),
#             'cv_score': cross_val_score(ensemble_model, X_train, y_train, cv=5, scoring='r2').mean()
#         }
#         results['Strategy 5: Ensemble'] = metrics_ensemble
#         models['Strategy 5'] = ensemble_model
#
#     print("\n=== STRATEGY COMPARISON ===")
#     comparison_df = pd.DataFrame(results).T.sort_values('r2', ascending=False)
#     print(comparison_df)
#
#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#     for ax, metric, color in zip(axs, ['r2', 'rmse', 'mae'], ['skyblue', 'salmon', 'lightgreen']):
#         ax.bar(comparison_df.index, comparison_df[metric], color=color)
#         ax.set_title(f'{metric.upper()} Comparison')
#         ax.set_ylabel(metric.upper())
#         ax.set_xticklabels(comparison_df.index, rotation=45)
#
#     plt.tight_layout()
#     plt.show()
#
#     best_strategy = comparison_df.index[0]
#     best_model = models[best_strategy.split(':')[0]]
#     best_metrics = results[best_strategy]
#
#     print(f"\n=== BEST STRATEGY: {best_strategy} ===")
#     for key, value in best_metrics.items():
#         print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
#
#     joblib.dump(best_model, 'results/best_enhanced_lasso_strat23.joblib')
#     print("Saved best model to results/best_enhanced_lasso_strat23.joblib")
#
#     try:
#         print("\nRunning SHAP diagnostics for best model...")
#         explainer = shap.Explainer(best_model.predict, X_test)
#         shap_values = explainer(X_test)
#         shap.summary_plot(shap_values, X_test, show=True)
#     except Exception as e:
#         print(f"SHAP diagnostics skipped: {e}")
#
#     return best_model, best_metrics
#
#
# def strategy_feature_selection(X_train, X_test, y_train, y_test, cat_features, num_features):
#     preprocessor = ColumnTransformer([
#         ('num', Pipeline([
#             ('imputer', KNNImputer(n_neighbors=3)),
#             ('scaler', StandardScaler())
#         ]), num_features),
#         ('cat', Pipeline([
#             ('imputer', SimpleImputer(strategy='most_frequent')),
#             ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
#         ]), cat_features)
#     ])
#
#     X_train_prep = preprocessor.fit_transform(X_train)
#     X_test_prep = preprocessor.transform(X_test)
#
#     selector = SelectKBest(score_func=f_regression, k=min(20, X_train_prep.shape[1]))
#     X_train_sel = selector.fit_transform(X_train_prep, y_train)
#     X_test_sel = selector.transform(X_test_prep)
#
#     model = LassoCV(alphas=np.logspace(-4, 1, 40), cv=5, max_iter=3000)
#     model.fit(X_train_sel, y_train)
#
#     y_pred = model.predict(X_test_sel)
#     metrics = {
#         'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
#         'r2': r2_score(y_test, y_pred),
#         'mae': mean_absolute_error(y_test, y_pred),
#         'cv_score': cross_val_score(model, X_train_sel, y_train, scoring='r2', cv=5).mean()
#     }
#
#     pipeline = Pipeline([
#         ('preprocessor', preprocessor),
#         ('selector', selector),
#         ('model', model)
#     ])
#
#     return pipeline, metrics
#
#
# def strategy_target_transformation(X_train, X_test, y_train, y_test, cat_features, num_features):
#     transformations = {
#         'log': np.log1p,
#         'sqrt': np.sqrt,
#         'boxcox': lambda x: stats.boxcox(x + 1)[0]
#     }
#
#     best_model = None
#     best_metrics = None
#     best_score = -np.inf
#     best_name = None
#
#     for name, transform_func in transformations.items():
#         try:
#             if name == 'boxcox':
#                 y_train_trans, lambda_val = stats.boxcox(y_train + 1)
#                 y_test_trans = stats.boxcox(y_test + 1, lmbda=lambda_val)
#             else:
#                 y_train_trans = transform_func(y_train)
#                 y_test_trans = transform_func(y_test)
#
#             preprocessor = ColumnTransformer([
#                 ('num', Pipeline([
#                     ('imputer', KNNImputer(n_neighbors=5)),
#                     ('quantile', QuantileTransformer(output_distribution='normal')),
#                     ('scaler', StandardScaler())
#                 ]), num_features),
#                 ('cat', Pipeline([
#                     ('imputer', SimpleImputer(strategy='most_frequent')),
#                     ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
#                 ]), cat_features)
#             ])
#
#             model = Pipeline([
#                 ('preprocessor', preprocessor),
#                 ('regressor', LassoCV(alphas=np.logspace(-4, 1, 40), cv=5, max_iter=3000))
#             ])
#
#             model.fit(X_train, y_train_trans)
#             y_pred_trans = model.predict(X_test)
#
#             if name == 'log':
#                 y_pred = np.expm1(y_pred_trans)
#             elif name == 'sqrt':
#                 y_pred = y_pred_trans**2
#             elif name == 'boxcox':
#                 y_pred = stats.inv_boxcox(y_pred_trans, lambda_val) - 1
#
#             y_pred = np.maximum(y_pred, 0)
#             r2 = r2_score(y_test, y_pred)
#
#             if r2 > best_score:
#                 best_score = r2
#                 best_model = model
#                 best_name = name
#                 best_metrics = {
#                     'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
#                     'r2': r2,
#                     'mae': mean_absolute_error(y_test, y_pred),
#                     'transformation': name,
#                     'cv_score': cross_val_score(model, X_train, y_train, scoring='r2', cv=5).mean()
#                 }
#         except Exception as e:
#             print(f"Transformation {name} failed: {e}")
#             continue
#
#     print(f"\nBest transformation: {best_name}")
#     return best_model, best_metrics

#########################0.54 I think so

"""
Recursive Feature Elimination Strategy for Lasso Regression
with Cross-Validation R² Score Display
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')


def strategy_recursive_feature_elimination(X_train, X_test, y_train, y_test, cat_features, num_features):
    """
    Strategy: Use Recursive Feature Elimination with Cross-Validation
    Includes CV R² calculation and console output
    """
    print("Starting Recursive Feature Elimination Strategy...")
    print("="*50)

    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ]), num_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]), cat_features)
    ])

    print("Preprocessing data...")
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    print(f"Features after preprocessing: {X_train_prep.shape[1]}")

    # Use RFECV to automatically select optimal number of features
    print("\nInitializing Lasso estimator for RFE...")
    estimator = LassoCV(alphas=np.logspace(-4, 1, 20), cv=3, max_iter=2000)

    print("Running Recursive Feature Elimination with Cross-Validation...")
    selector = RFECV(
        estimator=estimator,
        step=0.1,  # Remove 10% of features at each iteration
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1  # Show progress
    )

    print("Fitting RFECV selector...")
    X_train_sel = selector.fit_transform(X_train_prep, y_train)
    X_test_sel = selector.transform(X_test_prep)

    print(f"\nRFECV Results:")
    print(f"Optimal number of features: {selector.n_features_}")
    print(f"Features selected: {X_train_sel.shape[1]}")
    print(f"Best CV score during RFE: {selector.grid_scores_.max():.4f}")

    # Train final model on selected features
    print("\nTraining final Lasso model on selected features...")
    final_model = LassoCV(alphas=np.logspace(-4, 1, 40), cv=5, max_iter=3000)
    final_model.fit(X_train_sel, y_train)

    print(f"Final model optimal alpha: {final_model.alpha_:.6f}")
    print(f"Non-zero coefficients: {np.sum(final_model.coef_ != 0)}")

    # Calculate cross-validation R² score
    print("\nCalculating Cross-Validation R² Score...")
    cv_scores = cross_val_score(final_model, X_train_sel, y_train,
                               scoring='r2', cv=5, n_jobs=-1)

    print("Individual CV fold R² scores:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score:.4f}")

    cv_r2_mean = cv_scores.mean()
    cv_r2_std = cv_scores.std()

    print(f"\nCV R² Score: {cv_r2_mean:.4f} (+/- {cv_r2_std * 2:.4f})")

    # Make predictions and calculate test metrics
    print("\nMaking predictions on test set...")
    y_pred = final_model.predict(X_test_sel)

    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)

    print(f"\nTest Set Performance:")
    print(f"R² Score: {test_r2:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAE: {test_mae:.4f}")

    metrics = {
        'rmse': test_rmse,
        'r2': test_r2,
        'mae': test_mae,
        'cv_r2': cv_r2_mean,
        'cv_r2_std': cv_r2_std,
        'n_features_selected': selector.n_features_,
        'optimal_features': X_train_sel.shape[1],
        'best_rfe_score': selector.grid_scores_.max(),
        'final_alpha': final_model.alpha_,
        'non_zero_coefs': np.sum(final_model.coef_ != 0)
    }

    # Create complete pipeline for consistency
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('model', final_model)
    ])

    print("\n" + "="*50)
    print("Recursive Feature Elimination Strategy Complete!")
    print("="*50)

    return pipeline, metrics


# Example usage function
def run_rfe_strategy(X_train, X_test, y_train, y_test, categorical_features, numerical_features):
    """
    Wrapper function to run the RFE strategy
    """
    model, metrics = strategy_recursive_feature_elimination(
        X_train, X_test, y_train, y_test,
        categorical_features, numerical_features
    )

    print("\nFINAL SUMMARY:")
    print("-" * 30)
    print(f"Cross-Validation R²: {metrics['cv_r2']:.4f} ± {metrics['cv_r2_std']:.4f}")
    print(f"Test R²: {metrics['r2']:.4f}")
    print(f"Test RMSE: {metrics['rmse']:.4f}")
    print(f"Test MAE: {metrics['mae']:.4f}")
    print(f"Features Selected: {metrics['n_features_selected']}")
    print(f"Non-zero Coefficients: {metrics['non_zero_coefs']}")

    return model, metrics