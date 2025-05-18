
# Feature importance analysis
def analyze_feature_importance(model_name, model):
    print(f"\n--- Feature Importance Analysis for {model_name} ---")

    if model_name in ['Random Forest', 'Gradient Boosting']:
        try:
            # Get feature names after preprocessing
            # First, fit the preprocessor to get the transformed feature names
            preprocessor.fit(X_train)

            # Get feature names after one-hot encoding
            cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
            feature_names = list(numerical_features) + list(cat_features)

            # Get importances
            importances = model.named_steps['model'].feature_importances_

            # Ensure we have the right number of features
            if len(importances) == len(feature_names):
                # Create DataFrame for feature importance
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False)

                # Plot feature importance
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))  # Top 20 features
                plt.title(f'Top 20 Feature Importance - {model_name}')
                plt.tight_layout()
                plt.show()

                print("Top 10 most important features:")
                print(importance_df.head(10))
            else:
                print(f"Feature names length ({len(feature_names)}) doesn't match importances length ({len(importances)})")

        except Exception as e:
            print(f"Error in feature importance analysis: {str(e)}")

    elif model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet']:
        try:
            # Get feature names after preprocessing
            preprocessor.fit(X_train)

            # Get feature names after one-hot encoding
            cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
            feature_names = list(numerical_features) + list(cat_features)

            # Get coefficients
            coefficients = model.named_steps['model'].coef_

            # Ensure we have the right number of features
            if len(coefficients) == len(feature_names):
                # Create DataFrame for coefficients
                coef_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': coefficients
                })

                # Sort by absolute value
                coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
                coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)

                # Plot coefficients
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Coefficient', y='Feature', data=coef_df.head(20))  # Top 20 features
                plt.title(f'Top 20 Feature Coefficients - {model_name}')
                plt.axvline(x=0, color='r', linestyle='-')
                plt.tight_layout()
                plt.show()

                print("Top 10 features by coefficient magnitude:")
                print(coef_df[['Feature', 'Coefficient']].head(10))
            else:
                print(f"Feature names length ({len(feature_names)}) doesn't match coefficients length ({len(coefficients)})")

        except Exception as e:
            print(f"Error in coefficient analysis: {str(e)}")

    else:
        print(f"Feature importance analysis not implemented for {model_name}")

# Plot feature importance for the best model
analyze_feature_importance(best_model_name, best_model)
