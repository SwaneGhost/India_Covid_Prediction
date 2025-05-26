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

#
# """
# Additional Enhancement Strategies for Lasso Regression
# Building upon the existing strategies with advanced techniques
# """
#
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LassoCV, ElasticNetCV, RidgeCV
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.preprocessing import PolynomialFeatures, RobustScaler, StandardScaler, OrdinalEncoder, QuantileTransformer
# from sklearn.feature_selection import SelectFromModel, RFE, RFECV, SelectKBest, f_regression
# from sklearn.decomposition import PCA
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer, KNNImputer
# from sklearn.base import clone
# from scipy import stats
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')
#
#
# def strategy_elastic_net_optimization(X_train, X_test, y_train, y_test, cat_features, num_features):
#     """
#     Strategy 4: Use Elastic Net with optimized alpha and l1_ratio
#     Combines Ridge (L2) and Lasso (L1) regularization
#     """
#     preprocessor = ColumnTransformer([
#         ('num', Pipeline([
#             ('imputer', KNNImputer(n_neighbors=5)),
#             ('scaler', RobustScaler())  # More robust to outliers than StandardScaler
#         ]), num_features),
#         ('cat', Pipeline([
#             ('imputer', SimpleImputer(strategy='most_frequent')),
#             ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
#         ]), cat_features)
#     ])
#
#     # Wider range of l1_ratio values (0 = Ridge, 1 = Lasso)
#     l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
#     alphas = np.logspace(-4, 2, 50)
#
#     model = Pipeline([
#         ('preprocessor', preprocessor),
#         ('regressor', ElasticNetCV(
#             alphas=alphas,
#             l1_ratio=l1_ratios,
#             cv=10,  # More folds for better validation
#             max_iter=5000,
#             selection='random'  # Can be faster for large datasets
#         ))
#     ])
#
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#
#     metrics = {
#         'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
#         'r2': r2_score(y_test, y_pred),
#         'mae': mean_absolute_error(y_test, y_pred),
#         'best_alpha': model.named_steps['regressor'].alpha_,
#         'best_l1_ratio': model.named_steps['regressor'].l1_ratio_
#     }
#
#     return model, metrics
#
#
# def strategy_polynomial_features(X_train, X_test, y_train, y_test, cat_features, num_features):
#     """
#     Strategy 5: Add polynomial features for non-linear relationships
#     """
#     preprocessor = ColumnTransformer([
#         ('num', Pipeline([
#             ('imputer', KNNImputer(n_neighbors=5)),
#             ('scaler', StandardScaler()),
#             ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
#         ]), num_features),
#         ('cat', Pipeline([
#             ('imputer', SimpleImputer(strategy='most_frequent')),
#             ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
#         ]), cat_features)
#     ])
#
#     # Use more aggressive regularization due to increased features
#     model = Pipeline([
#         ('preprocessor', preprocessor),
#         ('regressor', LassoCV(
#             alphas=np.logspace(-3, 2, 50),
#             cv=5,
#             max_iter=3000
#         ))
#     ])
#
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#
#     metrics = {
#         'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
#         'r2': r2_score(y_test, y_pred),
#         'mae': mean_absolute_error(y_test, y_pred),
#         'n_features': model.named_steps['preprocessor'].transform(X_train).shape[1]
#     }
#
#     return model, metrics
#
#
# def strategy_recursive_feature_elimination(X_train, X_test, y_train, y_test, cat_features, num_features):
#     """
#     Strategy 6: Use Recursive Feature Elimination with Cross-Validation
#     """
#     preprocessor = ColumnTransformer([
#         ('num', Pipeline([
#             ('imputer', KNNImputer(n_neighbors=5)),
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
#     # Use RFECV to automatically select optimal number of features
#     estimator = LassoCV(alphas=np.logspace(-4, 1, 20), cv=3, max_iter=2000)
#     selector = RFECV(
#         estimator=estimator,
#         step=0.1,  # Remove 10% of features at each iteration
#         cv=5,
#         scoring='r2',
#         n_jobs=-1
#     )
#
#     X_train_sel = selector.fit_transform(X_train_prep, y_train)
#     X_test_sel = selector.transform(X_test_prep)
#
#     final_model = LassoCV(alphas=np.logspace(-4, 1, 40), cv=5, max_iter=3000)
#     final_model.fit(X_train_sel, y_train)
#
#     y_pred = final_model.predict(X_test_sel)
#
#     metrics = {
#         'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
#         'r2': r2_score(y_test, y_pred),
#         'mae': mean_absolute_error(y_test, y_pred),
#         'n_features_selected': selector.n_features_,
#         'optimal_features': X_train_sel.shape[1]
#     }
#
#     # Create pipeline for consistency
#     pipeline = Pipeline([
#         ('preprocessor', preprocessor),
#         ('selector', selector),
#         ('model', final_model)
#     ])
#
#     return pipeline, metrics
#
#
# def strategy_ensemble_with_other_regressors(X_train, X_test, y_train, y_test, cat_features, num_features):
#     """
#     Strategy 7: Ensemble Lasso with other complementary regressors
#     """
#     from sklearn.ensemble import VotingRegressor
#
#     preprocessor = ColumnTransformer([
#         ('num', Pipeline([
#             ('imputer', KNNImputer(n_neighbors=5)),
#             ('scaler', StandardScaler())
#         ]), num_features),
#         ('cat', Pipeline([
#             ('imputer', SimpleImputer(strategy='most_frequent')),
#             ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
#         ]), cat_features)
#     ])
#
#     # Different models that complement Lasso
#     models = [
#         ('lasso', Pipeline([
#             ('prep', clone(preprocessor)),
#             ('reg', LassoCV(alphas=np.logspace(-4, 1, 30), cv=5, max_iter=3000))
#         ])),
#         ('elastic', Pipeline([
#             ('prep', clone(preprocessor)),
#             ('reg', ElasticNetCV(alphas=np.logspace(-4, 1, 20), l1_ratio=[0.1, 0.5, 0.9], cv=5))
#         ])),
#         ('ridge', Pipeline([
#             ('prep', clone(preprocessor)),
#             ('reg', RidgeCV(alphas=np.logspace(-4, 2, 30), cv=5))
#         ]))
#     ]
#
#     ensemble = VotingRegressor(estimators=models, n_jobs=-1)
#     ensemble.fit(X_train, y_train)
#     y_pred = ensemble.predict(X_test)
#
#     metrics = {
#         'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
#         'r2': r2_score(y_test, y_pred),
#         'mae': mean_absolute_error(y_test, y_pred)
#     }
#
#     return ensemble, metrics
#
#
# def strategy_stacking_ensemble(X_train, X_test, y_train, y_test, cat_features, num_features):
#     """
#     Strategy 8: Stacking ensemble with Lasso as meta-learner
#     """
#     from sklearn.ensemble import StackingRegressor
#
#     preprocessor = ColumnTransformer([
#         ('num', Pipeline([
#             ('imputer', KNNImputer(n_neighbors=5)),
#             ('scaler', StandardScaler())
#         ]), num_features),
#         ('cat', Pipeline([
#             ('imputer', SimpleImputer(strategy='most_frequent')),
#             ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
#         ]), cat_features)
#     ])
#
#     # Base learners
#     base_models = [
#         ('rf', Pipeline([
#             ('prep', clone(preprocessor)),
#             ('reg', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
#         ])),
#         ('gb', Pipeline([
#             ('prep', clone(preprocessor)),
#             ('reg', GradientBoostingRegressor(n_estimators=50, random_state=42))
#         ])),
#         ('elastic', Pipeline([
#             ('prep', clone(preprocessor)),
#             ('reg', ElasticNetCV(alphas=np.logspace(-4, 1, 20), cv=3))
#         ]))
#     ]
#
#     # Meta-learner (Lasso)
#     meta_learner = Pipeline([
#         ('prep', preprocessor),
#         ('reg', LassoCV(alphas=np.logspace(-4, 1, 20), cv=3, max_iter=2000))
#     ])
#
#     stacking_model = StackingRegressor(
#         estimators=base_models,
#         final_estimator=meta_learner,
#         cv=3,
#         n_jobs=-1
#     )
#
#     stacking_model.fit(X_train, y_train)
#     y_pred = stacking_model.predict(X_test)
#
#     metrics = {
#         'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
#         'r2': r2_score(y_test, y_pred),
#         'mae': mean_absolute_error(y_test, y_pred)
#     }
#
#     return stacking_model, metrics
#
#
# def strategy_adaptive_lasso(X_train, X_test, y_train, y_test, cat_features, num_features):
#     """
#     Strategy 9: Adaptive Lasso - weights features based on initial OLS/Ridge estimates
#     """
#     from sklearn.linear_model import Ridge
#
#     preprocessor = ColumnTransformer([
#         ('num', Pipeline([
#             ('imputer', KNNImputer(n_neighbors=5)),
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
#     # Step 1: Get initial weights from Ridge regression
#     ridge = Ridge(alpha=1.0)
#     ridge.fit(X_train_prep, y_train)
#     weights = 1.0 / (np.abs(ridge.coef_) + 1e-8)  # Avoid division by zero
#
#     # Step 2: Apply adaptive weights to features
#     X_train_weighted = X_train_prep * weights
#     X_test_weighted = X_test_prep * weights
#
#     # Step 3: Apply Lasso to weighted features
#     adaptive_lasso = LassoCV(alphas=np.logspace(-4, 1, 40), cv=5, max_iter=3000)
#     adaptive_lasso.fit(X_train_weighted, y_train)
#
#     y_pred = adaptive_lasso.predict(X_test_weighted)
#
#     metrics = {
#         'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
#         'r2': r2_score(y_test, y_pred),
#         'mae': mean_absolute_error(y_test, y_pred),
#         'n_nonzero_coefs': np.sum(adaptive_lasso.coef_ != 0)
#     }
#
#     # Create a custom pipeline for this approach
#     class AdaptiveLassoTransformer:
#         def init(self, weights):
#             self.weights = weights
#
#         def fit(self, X, y=None):
#             return self
#
#         def transform(self, X):
#             return X * self.weights
#
#     pipeline = Pipeline([
#         ('preprocessor', preprocessor),
#         ('adaptive_weights', AdaptiveLassoTransformer(weights)),
#         ('lasso', adaptive_lasso)
#     ])
#
#     return pipeline, metrics
#
#
# def enhanced_lasso_all_strategies(X_train, X_test, y_train, y_test, categorical_features, numerical_features):
#     """
#     Run all enhancement strategies and compare results
#     """
#     # Import the original strategies from your existing file
#     import sys
#     import os
#     sys.path.append(os.path.join(os.getcwd(), 'Code'))
#
#     try:
#         from enhance_lasso_strategies import strategy_feature_selection, strategy_target_transformation
#     except ImportError:
#         # Fallback: define the strategies locally if import fails
#         print("Warning: Could not import original strategies. Using local implementations.")
#
#         def strategy_feature_selection(X_train, X_test, y_train, y_test, cat_features, num_features):
#             from sklearn.impute import SimpleImputer, KNNImputer
#             from sklearn.preprocessing import StandardScaler, OrdinalEncoder
#             from sklearn.compose import ColumnTransformer
#             from sklearn.pipeline import Pipeline
#             from sklearn.feature_selection import SelectKBest, f_regression
#             from sklearn.linear_model import LassoCV
#             from sklearn.model_selection import cross_val_score
#
#             preprocessor = ColumnTransformer([
#                 ('num', Pipeline([
#                     ('imputer', KNNImputer(n_neighbors=3)),
#                     ('scaler', StandardScaler())
#                 ]), num_features),
#                 ('cat', Pipeline([
#                     ('imputer', SimpleImputer(strategy='most_frequent')),
#                     ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
#                 ]), cat_features)
#             ])
#
#             X_train_prep = preprocessor.fit_transform(X_train)
#             X_test_prep = preprocessor.transform(X_test)
#
#             selector = SelectKBest(score_func=f_regression, k=min(50, X_train_prep.shape[1]))  # Increased from 20
#             X_train_sel = selector.fit_transform(X_train_prep, y_train)
#             X_test_sel = selector.transform(X_test_prep)
#
#             model = LassoCV(alphas=np.logspace(-4, 1, 40), cv=5, max_iter=3000)
#             model.fit(X_train_sel, y_train)
#
#             y_pred = model.predict(X_test_sel)
#             metrics = {
#                 'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
#                 'r2': r2_score(y_test, y_pred),
#                 'mae': mean_absolute_error(y_test, y_pred),
#                 'cv_score': cross_val_score(model, X_train_sel, y_train, scoring='r2', cv=5).mean()
#             }
#
#             pipeline = Pipeline([
#                 ('preprocessor', preprocessor),
#                 ('selector', selector),
#                 ('model', model)
#             ])
#
#             return pipeline, metrics
#
#         def strategy_target_transformation(X_train, X_test, y_train, y_test, cat_features, num_features):
#             from sklearn.impute import SimpleImputer, KNNImputer
#             from sklearn.preprocessing import StandardScaler, OrdinalEncoder, QuantileTransformer
#             from sklearn.compose import ColumnTransformer
#             from sklearn.pipeline import Pipeline
#             from sklearn.linear_model import LassoCV
#             from sklearn.model_selection import cross_val_score
#             from scipy import stats
#
#             transformations = {
#                 'log': np.log1p,
#                 'sqrt': np.sqrt,
#                 'boxcox': lambda x: stats.boxcox(x + 1)[0]
#             }
#
#             best_model = None
#             best_metrics = None
#             best_score = -np.inf
#             best_name = None
#
#             for name, transform_func in transformations.items():
#                 try:
#                     if name == 'boxcox':
#                         y_train_trans, lambda_val = stats.boxcox(y_train + 1)
#                         y_test_trans = stats.boxcox(y_test + 1, lmbda=lambda_val)
#                     else:
#                         y_train_trans = transform_func(y_train)
#                         y_test_trans = transform_func(y_test)
#
#                     preprocessor = ColumnTransformer([
#                         ('num', Pipeline([
#                             ('imputer', KNNImputer(n_neighbors=5)),
#                             ('quantile', QuantileTransformer(output_distribution='normal')),
#                             ('scaler', StandardScaler())
#                         ]), num_features),
#                         ('cat', Pipeline([
#                             ('imputer', SimpleImputer(strategy='most_frequent')),
#                             ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
#                         ]), cat_features)
#                     ])
#
#                     model = Pipeline([
#                         ('preprocessor', preprocessor),
#                         ('regressor', LassoCV(alphas=np.logspace(-4, 1, 40), cv=5, max_iter=3000))
#                     ])
#
#                     model.fit(X_train, y_train_trans)
#                     y_pred_trans = model.predict(X_test)
#
#                     if name == 'log':
#                         y_pred = np.expm1(y_pred_trans)
#                     elif name == 'sqrt':
#                         y_pred = y_pred_trans**2
#                     elif name == 'boxcox':
#                         # Manual inverse Box-Cox transformation
#                         if lambda_val == 0:
#                             y_pred = np.exp(y_pred_trans) - 1
#                         else:
#                             y_pred = np.power((lambda_val * y_pred_trans + 1), (1.0 / lambda_val)) - 1
#
#                     y_pred = np.maximum(y_pred, 0)
#                     r2 = r2_score(y_test, y_pred)
#
#                     if r2 > best_score:
#                         best_score = r2
#                         best_model = model
#                         best_name = name
#                         best_metrics = {
#                             'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
#                             'r2': r2,
#                             'mae': mean_absolute_error(y_test, y_pred),
#                             'transformation': name,
#                             'cv_score': cross_val_score(model, X_train, y_train, scoring='r2', cv=5).mean()
#                         }
#                 except Exception as e:
#                     print(f"Transformation {name} failed: {e}")
#                     continue
#
#             return best_model, best_metrics
#
#     results = {}
#     models = {}
#
#     strategies = {
#         'Feature Selection': lambda: strategy_feature_selection(X_train, X_test, y_train, y_test, categorical_features, numerical_features),
#         'Target Transformation': lambda: strategy_target_transformation(X_train, X_test, y_train, y_test, categorical_features, numerical_features),
#         'Elastic Net Optimization': lambda: strategy_elastic_net_optimization(X_train, X_test, y_train, y_test, categorical_features, numerical_features),
#         'Polynomial Features': lambda: strategy_polynomial_features(X_train, X_test, y_train, y_test, categorical_features, numerical_features),
#         'Recursive Feature Elimination': lambda: strategy_recursive_feature_elimination(X_train, X_test, y_train, y_test, categorical_features, numerical_features),
#         'Multi-Model Ensemble': lambda: strategy_ensemble_with_other_regressors(X_train, X_test, y_train, y_test, categorical_features, numerical_features),
#         'Stacking Ensemble': lambda: strategy_stacking_ensemble(X_train, X_test, y_train, y_test, categorical_features, numerical_features),
#         'Adaptive Lasso': lambda: strategy_adaptive_lasso(X_train, X_test, y_train, y_test, categorical_features, numerical_features)
#     }
#
#     print("Running all enhancement strategies...")
#     for name, strategy_func in strategies.items():
#         try:
#             print(f"\nRunning: {name}")
#             model, metrics = strategy_func()
#             results[name] = metrics
#             models[name] = model
#             print(f"R² Score: {metrics['r2']:.4f}")
#         except Exception as e:
#             print(f"Strategy {name} failed: {e}")
#             continue
#
#     # Create comparison
#     comparison_df = pd.DataFrame(results).T.sort_values('r2', ascending=False)
#     print("\n" + "="*60)
#     print("COMPREHENSIVE STRATEGY COMPARISON")
#     print("="*60)
#     print(comparison_df.round(4))
#
#     # Find best strategy
#     best_strategy = comparison_df.index[0]
#     best_model = models[best_strategy]
#     best_metrics = results[best_strategy]
#
#     print(f"\n🏆 BEST STRATEGY: {best_strategy}")
#     print(f"R² Score: {best_metrics['r2']:.4f}")
#     print(f"RMSE: {best_metrics['rmse']:.4f}")
#     print(f"MAE: {best_metrics['mae']:.4f}")
#
#     return best_model, best_metrics, comparison_df


######################## only recursive elimination


import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.base import clone
import warnings

warnings.filterwarnings('ignore')


def strategy_recursive_feature_elimination_with_cv(X_train, X_test, y_train, y_test, cat_features, num_features,
                                                   min_features=10, max_features=30, step_size=1):
    """
    Recursive Feature Elimination with Cross-Validation for Lasso Regression
    Includes comprehensive CV scoring to evaluate model performance

    Parameters:
    - min_features: minimum number of features to keep
    - max_features: maximum number of features to consider
    - step_size: number of features to remove at each iteration
    """
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

    # Preprocess the data
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    print(f"Total features after preprocessing: {X_train_prep.shape[1]}")

    # Use less aggressive regularization for initial feature ranking
    estimator = LassoCV(alphas=np.logspace(-6, 0, 30), cv=3, max_iter=3000)

    # More conservative RFECV settings
    selector = RFECV(
        estimator=estimator,
        step=step_size,  # Remove fewer features at each iteration
        cv=5,
        scoring='r2',
        min_features_to_select=min_features,  # Ensure minimum features
        n_jobs=-1
    )

    # Apply feature selection
    X_train_sel = selector.fit_transform(X_train_prep, y_train)
    X_test_sel = selector.transform(X_test_prep)

    print(f"Features selected by RFECV: {X_train_sel.shape[1]}")

    # If RFECV selected too few features, use a fixed number approach
    if X_train_sel.shape[1] < min_features:
        print(f"RFECV selected too few features. Using top {max_features} features instead.")
        from sklearn.feature_selection import RFE

        # Use RFE with fixed number of features
        fixed_selector = RFE(
            estimator=LassoCV(alphas=np.logspace(-6, 0, 20), cv=3, max_iter=2000),
            n_features_to_select=max_features,
            step=1
        )

        X_train_sel = fixed_selector.fit_transform(X_train_prep, y_train)
        X_test_sel = fixed_selector.transform(X_test_prep)
        selector = fixed_selector  # Use this for the pipeline
        print(f"Fixed RFE selected: {X_train_sel.shape[1]} features")

    # Final model with selected features - use lighter regularization
    final_model = LassoCV(alphas=np.logspace(-6, 0, 50), cv=5, max_iter=3000)
    final_model.fit(X_train_sel, y_train)

    # Make predictions
    y_pred = final_model.predict(X_test_sel)

    # Cross-validation scoring on training data with selected features
    cv_scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    cv_results = cross_validate(
        final_model,
        X_train_sel,
        y_train,
        cv=5,
        scoring=cv_scoring,
        return_train_score=True
    )

    # Calculate CV metrics
    cv_metrics = {
        'cv_r2_mean': cv_results['test_r2'].mean(),
        'cv_r2_std': cv_results['test_r2'].std(),
        'cv_rmse_mean': np.sqrt(-cv_results['test_neg_mean_squared_error'].mean()),
        'cv_rmse_std': np.sqrt(cv_results['test_neg_mean_squared_error'].var()),
        'cv_mae_mean': -cv_results['test_neg_mean_absolute_error'].mean(),
        'cv_mae_std': cv_results['test_neg_mean_absolute_error'].std(),
        'cv_train_r2_mean': cv_results['train_r2'].mean(),
        'cv_train_r2_std': cv_results['train_r2'].std(),
    }

    # Test set metrics
    test_metrics = {
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'test_r2': r2_score(y_test, y_pred),
        'test_mae': mean_absolute_error(y_test, y_pred),
    }

    # Feature selection metrics
    feature_metrics = {
        'n_features_selected': X_train_sel.shape[1],  # Use actual shape instead of selector.n_features_
        'total_features': X_train_prep.shape[1],
        'feature_selection_ratio': X_train_sel.shape[1] / X_train_prep.shape[1],
        'best_alpha': final_model.alpha_,
        'n_nonzero_coefs': np.sum(final_model.coef_ != 0)
    }

    # Combine all metrics
    all_metrics = {**cv_metrics, **test_metrics, **feature_metrics}

    # Create final pipeline for consistency
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('model', final_model)
    ])

    return pipeline, all_metrics


def display_results(metrics):
    """
    Display the results in a nice formatted way
    """
    print("=" * 60)
    print("RECURSIVE FEATURE ELIMINATION RESULTS")
    print("=" * 60)

    print("\n📊 CROSS-VALIDATION PERFORMANCE:")
    print(f"  R² Score: {metrics['cv_r2_mean']:.4f} (± {metrics['cv_r2_std']:.4f})")
    print(f"  RMSE:     {metrics['cv_rmse_mean']:.4f} (± {metrics['cv_rmse_std']:.4f})")
    print(f"  MAE:      {metrics['cv_mae_mean']:.4f} (± {metrics['cv_mae_std']:.4f})")

    print(f"\n  Training R² (CV): {metrics['cv_train_r2_mean']:.4f} (± {metrics['cv_train_r2_std']:.4f})")

    print("\n🎯 TEST SET PERFORMANCE:")
    print(f"  R² Score: {metrics['test_r2']:.4f}")
    print(f"  RMSE:     {metrics['test_rmse']:.4f}")
    print(f"  MAE:      {metrics['test_mae']:.4f}")

    print("\n🔍 FEATURE SELECTION INFO:")
    print(f"  Features Selected: {metrics['n_features_selected']}/{metrics['total_features']}")
    print(f"  Selection Ratio:   {metrics['feature_selection_ratio']:.2%}")
    print(f"  Non-zero Coeffs:   {metrics['n_nonzero_coefs']}")
    print(f"  Best Alpha:        {metrics['best_alpha']:.6f}")

    # Check for overfitting
    train_test_gap = metrics['cv_train_r2_mean'] - metrics['cv_r2_mean']
    print(f"\n📈 OVERFITTING CHECK:")
    print(f"  Train-CV Gap: {train_test_gap:.4f}")
    if train_test_gap > 0.1:
        print("  ⚠️  Potential overfitting detected")
    elif train_test_gap < 0.02:
        print("  ✅ Good generalization")
    else:
        print("  ⚠️  Moderate overfitting")


# Example usage function
def run_rfe_analysis(X_train, X_test, y_train, y_test, categorical_features, numerical_features,
                     min_features=10, max_features=25):
    """
    Run the complete RFE analysis and display results

    Parameters:
    - min_features: minimum number of features to keep (default: 10)
    - max_features: maximum number of features to consider (default: 25)
    """
    print("Running Recursive Feature Elimination with Cross-Validation...")

    model, metrics = strategy_recursive_feature_elimination_with_cv(
        X_train, X_test, y_train, y_test,
        categorical_features, numerical_features,
        min_features=min_features,
        max_features=max_features
    )

    display_results(metrics)

    return model, metrics


def get_selected_feature_names(model, feature_names):
    """
    Get the names of selected features from the trained model
    Useful for understanding which demographic/socioeconomic factors were selected
    """
    try:
        # Get feature names after preprocessing
        preprocessor = model.named_steps['preprocessor']
        selector = model.named_steps['selector']

        # Get feature names after preprocessing (this is tricky with ColumnTransformer)
        feature_names_transformed = []

        # Add numerical feature names
        num_features = preprocessor.named_transformers_['num'].get_feature_names_out() if hasattr(
            preprocessor.named_transformers_['num'], 'get_feature_names_out') else feature_names['numerical']
        feature_names_transformed.extend(num_features)

        # Add categorical feature names
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out() if hasattr(
            preprocessor.named_transformers_['cat'], 'get_feature_names_out') else feature_names['categorical']
        feature_names_transformed.extend(cat_features)

        # Get selected feature indices
        selected_features = selector.get_support(indices=True)
        selected_feature_names = [feature_names_transformed[i] for i in selected_features]

        print("\n🎯 SELECTED FEATURES:")
        for i, feature in enumerate(selected_feature_names, 1):
            print(f"  {i:2d}. {feature}")

        return selected_feature_names
    except Exception as e:
        print(f"Could not extract feature names: {e}")
        return None


def analyze_feature_importance(model, X_train_sample=None):
    """
    Analyze the importance of selected features
    """
    try:
        final_model = model.named_steps['model']
        coefficients = final_model.coef_

        print("\n📊 FEATURE COEFFICIENTS:")
        coef_abs = np.abs(coefficients)
        sorted_indices = np.argsort(coef_abs)[::-1]  # Sort by absolute value, descending

        for i, idx in enumerate(sorted_indices[:15]):  # Show top 15
            if coefficients[idx] != 0:
                print(f"  {i + 1:2d}. Feature {idx:2d}: {coefficients[idx]:8.4f} (|{coef_abs[idx]:.4f}|)")

    except Exception as e:
        print(f"Could not analyze feature importance: {e}")