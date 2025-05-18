# Train-test split with engineered features
X_train, X_test, y_train, y_test = train_test_split(X_engineered, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Visualize correlations of new engineered features with target
numeric_engineered = X_engineered.select_dtypes(include=['number']).columns
# Merge with the target for correlation calculation
correlation_data = pd.concat([X_engineered[numeric_engineered], pd.DataFrame({'cum_positive_cases': y})], axis=1)
correlation = correlation_data.corr()['cum_positive_cases'].sort_values(ascending=False)
print("\nTop 15 correlations with cumulative positive cases (engineered features):")
print(correlation.head(15))

# Visualize top engineered feature correlations
plt.figure(figsize=(14, 10))
top_corr = correlation.head(15)
correlation_df = pd.DataFrame(top_corr).reset_index()
correlation_df.columns = ['Feature', 'Correlation']
sns.barplot(x='Correlation', y='Feature', data=correlation_df)
plt.title('Top 15 Engineered Features: Correlation with Cumulative Positive Cases')
plt.axvline(x=0, color='r', linestyle='--')
plt.tight_layout()
plt.show()

# Define models to compare
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'ElasticNet': ElasticNet(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Compare models with engineered features
print("\n--- Model Comparison Using Engineered Demographic & Socioeconomic Features ---")
model_results = compare_models(models, X_train, X_test, y_train, y_test)

# Visualize model comparison
model_names = list(model_results.keys())
rmse_values = [model_results[model]['rmse'] for model in model_names]
r2_values = [model_results[model]['r2'] for model in model_names]

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
bars = plt.bar(model_names, rmse_values, color='skyblue')
plt.title('RMSE by Model with Engineered Features (Lower is Better)')
plt.xticks(rotation=45)
plt.ylabel('RMSE')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{bar.get_height():.2f}', ha='center')

plt.subplot(1, 2, 2)
bars = plt.bar(model_names, r2_values, color='lightgreen')
plt.title('R² by Model with Engineered Features (Higher is Better)')
plt.xticks(rotation=45)
plt.ylabel('R²')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{bar.get_height():.4f}', ha='center')

plt.tight_layout()
plt.show()

# Identify the best performing model based on R²
best_model_name = max(model_results, key=lambda x: model_results[x]['r2'])
print(f"\nBest performing model: {best_model_name} with R² = {model_results[best_model_name]['r2']:.4f}")

# Feature importance analysis for the best model
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print("\n--- Feature Importance Analysis ---")
    best_model = model_results[best_model_name]['model']

    try:
        # Extract feature names and importances
        # First, fit the preprocessor to get transformed feature names
        preprocessor.fit(X_train)

        # Get feature importances from the model
        feature_importances = best_model.named_steps['model'].feature_importances_

        # Get feature names
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_cols = cat_encoder.get_feature_names_out(categorical_features)

        all_feature_names = list(numerical_features) + list(cat_cols)

        # Match importances with feature names
        if len(feature_importances) == len(all_feature_names):
            # Create DataFrame for feature importance
            importance_df = pd.DataFrame({
                'Feature': all_feature_names,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)

            # Plot top 20 features
            plt.figure(figsize=(14, 10))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
            plt.title(f'Top 20 Feature Importance - {best_model_name} with Engineered Features')
            plt.tight_layout()
            plt.show()

            print("Top 10 most important features:")
            print(importance_df.head(10))
        else:
            print(f"Feature names length ({len(all_feature_names)}) doesn't match importances length ({len(feature_importances)})")
    except Exception as e:
        print(f"Error in feature importance analysis: {str(e)}")

# Save the best model
from joblib import dump
model_filename = f"{best_model_name.replace(' ', '_').lower()}_covid_prediction.joblib"
dump(model_results[best_model_name]['model'], model_filename)
print(f"\nBest model saved as: {model_filename}")

print("\n=== Conclusion ===")
print(f"The {best_model_name} with engineered demographic and socioeconomic features performed best")
print(f"with an R² of {model_results[best_model_name]['r2']:.4f}.")
print("The feature engineering significantly enriched the predictive power of the demographic data.")