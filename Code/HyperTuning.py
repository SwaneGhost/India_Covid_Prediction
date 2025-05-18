
# Hyperparameter tuning for all models
print("\n=== Hyperparameter Tuning for All Models ===")

def tune_all_models():
    tuned_models = {}
    tuning_results = {}

    # Linear Regression has no hyperparameters to tune
    print("\n--- Linear Regression ---")
    print("No hyperparameters to tune.")
    tuned_models['Linear Regression'] = model_results['Linear Regression']['model']
    tuning_results['Linear Regression'] = {
        'best_params': {},
        'rmse': model_results['Linear Regression']['rmse'],
        'r2': model_results['Linear Regression']['r2']
    }

    # Ridge Regression
    print("\n--- Ridge Regression ---")
    param_grid = {
        'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    }
    best_model, best_params, metrics = tune_model('Ridge Regression', Ridge(), param_grid)
    tuned_models['Ridge Regression'] = best_model
    tuning_results['Ridge Regression'] = {
        'best_params': best_params,
        'rmse': metrics['rmse'],
        'r2': metrics['r2']
    }

    # Lasso Regression
    print("\n--- Lasso Regression ---")
    param_grid = {
        'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        'model__max_iter': [1000, 3000, 5000]
    }
    best_model, best_params, metrics = tune_model('Lasso Regression', Lasso(), param_grid)
    tuned_models['Lasso Regression'] = best_model
    tuning_results['Lasso Regression'] = {
        'best_params': best_params,
        'rmse': metrics['rmse'],
        'r2': metrics['r2']
    }

    # ElasticNet
    print("\n--- ElasticNet ---")
    param_grid = {
        'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'model__max_iter': [1000, 3000]
    }
    best_model, best_params, metrics = tune_model('ElasticNet', ElasticNet(), param_grid)
    tuned_models['ElasticNet'] = best_model
    tuning_results['ElasticNet'] = {
        'best_params': best_params,
        'rmse': metrics['rmse'],
        'r2': metrics['r2']
    }

    # Random Forest
    print("\n--- Random Forest ---")
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }
    best_model, best_params, metrics = tune_model('Random Forest', RandomForestRegressor(random_state=42), param_grid)
    tuned_models['Random Forest'] = best_model
    tuning_results['Random Forest'] = {
        'best_params': best_params,
        'rmse': metrics['rmse'],
        'r2': metrics['r2']
    }

    # Gradient Boosting
    print("\n--- Gradient Boosting ---")
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__max_depth': [3, 5, 7],
        'model__min_samples_split': [2, 5],
        'model__subsample': [0.8, 1.0]
    }
    best_model, best_params, metrics = tune_model('Gradient Boosting', GradientBoostingRegressor(random_state=42), param_grid)
    tuned_models['Gradient Boosting'] = best_model
    tuning_results['Gradient Boosting'] = {
        'best_params': best_params,
        'rmse': metrics['rmse'],
        'r2': metrics['r2']
    }

    return tuned_models, tuning_results

def tune_model(name, base_model, param_grid):
    print(f"Tuning {name}...")

    # Create pipeline with preprocessor and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', base_model)
    ])

    # Set up grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )

    # Fit grid search
    grid_search.fit(X_train, y_train)

    # Best parameters and model
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    # Evaluate on test set
    y_pred = grid_search.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test set - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_, {'rmse': rmse, 'r2': r2, 'mae': mae}

# Perform hyperparameter tuning for all models
tuned_models, tuning_results = tune_all_models()

# Compare original and tuned model performance
print("\n=== Performance Comparison: Original vs. Tuned Models ===")
comparison_data = []

for name in model_names:
    original_rmse = model_results[name]['rmse']
    original_r2 = model_results[name]['r2']
    tuned_rmse = tuning_results[name]['rmse']
    tuned_r2 = tuning_results[name]['r2']
    improvement_rmse = ((original_rmse - tuned_rmse) / original_rmse) * 100
    improvement_r2 = ((tuned_r2 - original_r2) / original_r2) * 100 if original_r2 > 0 else float('inf')

    comparison_data.append({
        'Model': name,
        'Original RMSE': original_rmse,
        'Tuned RMSE': tuned_rmse,
        'RMSE Improvement (%)': improvement_rmse,
        'Original R²': original_r2,
        'Tuned R²': tuned_r2,
        'R² Improvement (%)': improvement_r2
    })

    print(f"\n{name}:")
    print(f"  Original: RMSE = {original_rmse:.2f}, R² = {original_r2:.4f}")
    print(f"  Tuned:    RMSE = {tuned_rmse:.2f}, R² = {tuned_r2:.4f}")
    print(f"  Improvement: RMSE = {improvement_rmse:.2f}%, R² = {improvement_r2:.2f}%")
    if tuning_results[name]['best_params']:
        print(f"  Best parameters: {tuning_results[name]['best_params']}")

# Create comparison dataframe
comparison_df = pd.DataFrame(comparison_data)

# Visualize RMSE comparison
plt.figure(figsize=(14, 6))
bar_width = 0.35
index = np.arange(len(model_names))

plt.subplot(1, 2, 1)
bars1 = plt.bar(index, comparison_df['Original RMSE'], bar_width, label='Original', color='skyblue')
bars2 = plt.bar(index + bar_width, comparison_df['Tuned RMSE'], bar_width, label='Tuned', color='lightgreen')
plt.xlabel('Model')
plt.ylabel('RMSE (Lower is Better)')
plt.title('RMSE Comparison: Original vs. Tuned Models')
plt.xticks(index + bar_width / 2, model_names, rotation=45)
plt.legend()

plt.subplot(1, 2, 2)
bars1 = plt.bar(index, comparison_df['Original R²'], bar_width, label='Original', color='skyblue')
bars2 = plt.bar(index + bar_width, comparison_df['Tuned R²'], bar_width, label='Tuned', color='lightgreen')
plt.xlabel('Model')
plt.ylabel('R² (Higher is Better)')
plt.title('R² Comparison: Original vs. Tuned Models')
plt.xticks(index + bar_width / 2, model_names, rotation=45)
plt.legend()

plt.tight_layout()
plt.show()

# Identify the best overall model
best_model_name = comparison_df.loc[comparison_df['Tuned R²'].idxmax(), 'Model']
best_model = tuned_models[best_model_name]
best_params = tuning_results[best_model_name]['best_params']

print(f"\n=== Best Overall Model ===")
print(f"Model: {best_model_name}")
print(f"Tuned R²: {tuning_results[best_model_name]['r2']:.4f}")
print(f"Tuned RMSE: {tuning_results[best_model_name]['rmse']:.2f}")
if best_params:
    print(f"Best parameters: {best_params}")


