from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint
import numpy as np
import joblib

def enhance_random_forest(X_train, X_test, y_train, y_test, preprocessor):
    # Apply log1p transformation to the target
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # Create a pipeline with preprocessing and RandomForest
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ])

    # Define parameter space for randomized search
    param_distributions = {
        'model__n_estimators': randint(100, 1000),
        'model__max_depth': [None] + list(np.arange(10, 100, 10)),
        'model__min_samples_split': randint(2, 20),
        'model__min_samples_leaf': randint(1, 20),
        'model__max_features': ['auto', 'sqrt', 'log2']
    }

    # Perform randomized search with cross-validation
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=50,
        scoring='r2',
        cv=5,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train_log)
    best_model = search.best_estimator_

    # Predict and reverse log transformation
    y_pred_log = best_model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test_log)

    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print("\n=== Enhanced Random Forest Performance ===")
    print("Best Parameters:", search.best_params_)
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")

    # Save the model
    joblib.dump(best_model, 'results/enhanced_random_forest.joblib')
    print("Enhanced model saved to results/enhanced_random_forest.joblib")

    return best_model
