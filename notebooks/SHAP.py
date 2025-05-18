# SHAP analysis for feature interpretation
def shap_analysis(model_name, model, X_train_sample, X_test_sample):
    print(f"\n--- SHAP Analysis for {model_name} ---")

    try:
        # Preprocess a sample of data
        X_train_processed = model.named_steps['preprocessor'].transform(X_train_sample)
        X_test_processed = model.named_steps['preprocessor'].transform(X_test_sample)

        # Get feature names after preprocessing
        preprocessor.fit(X_train)
        cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        feature_names = list(numerical_features) + list(cat_features)

        # Ensure processed data dimensions match expected feature names
        if X_test_processed.shape[1] != len(feature_names):
            print(f"Warning: Feature count mismatch. Processed data has {X_test_processed.shape[1]} features, but feature names list has {len(feature_names)}.")
            # Adjust feature names to match the processed data size
            if X_test_processed.shape[1] < len(feature_names):
                feature_names = feature_names[:X_test_processed.shape[1]]
            else:
                # If more features than names, extend feature names with generic names
                for i in range(len(feature_names), X_test_processed.shape[1]):
                    feature_names.append(f"feature_{i}")

        # Create SHAP explainer based on model type
        if model_name in ['Random Forest', 'Gradient Boosting']:
            explainer = shap.TreeExplainer(model.named_steps['model'])
            shap_values = explainer.shap_values(X_test_processed)

            # Summary plot
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, show=False)
            plt.title(f'SHAP Summary Plot - {model_name}')
            plt.tight_layout()
            plt.show()

        elif model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet']:
            explainer = shap.LinearExplainer(model.named_steps['model'], X_train_processed)
            shap_values = explainer.shap_values(X_test_processed)

            # Summary plot
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, show=False)
            plt.title(f'SHAP Summary Plot - {model_name}')
            plt.tight_layout()
            plt.show()

        else:
            # Fallback to KernelExplainer for other model types
            background = shap.sample(X_train_processed, 50)  # Sample for efficiency
            explainer = shap.KernelExplainer(model.named_steps['model'].predict, background)
            shap_values = explainer.shap_values(shap.sample(X_test_processed, 50))

            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, shap.sample(X_test_processed, 50), feature_names=feature_names, show=False)
            plt.title(f'SHAP Summary Plot - {model_name}')
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"Error in SHAP analysis: {str(e)}")

# Sample data for analysis (to avoid memory issues with SHAP)
X_train_sample = X_train.sample(min(1000, len(X_train)), random_state=42)
X_test_sample = X_test.sample(min(500, len(X_test)), random_state=42)

# Analyze feature importance
analyze_feature_importance(best_model_name, best_model)

# Perform SHAP analysis
shap_analysis(best_model_name, best_model, X_train_sample, X_test_sample)

# Make predictions on test data using the best model
y_pred = best_model.predict(X_test)

# Plot actual vs predicted values
plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Actual vs Predicted Cumulative Positive Cases - {best_model_name}')
plt.tight_layout()
plt.show()

# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(12, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()
plt.show()

# Histogram of residuals
plt.figure(figsize=(12, 6))
plt.hist(residuals, bins=50)
plt.axvline(x=0, color='r', linestyle='-')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.tight_layout()
plt.show()

# Create a results table with all models for easy comparison
results_table = pd.DataFrame({
    'Model': list(tuning_results.keys()),
    'Original R²': [model_results[model]['r2'] for model in model_results],
    'Tuned R²': [tuning_results[model]['r2'] for model in tuning_results],
    'Original RMSE': [model_results[model]['rmse'] for model in model_results],
    'Tuned RMSE': [tuning_results[model]['rmse'] for model in tuning_results],
})

# Sort by tuned R² (best performing first)
results_table = results_table.sort_values('Tuned R²', ascending=False).reset_index(drop=True)

# Display the results table
print("\n=== Model Performance Comparison ===")
print(results_table)