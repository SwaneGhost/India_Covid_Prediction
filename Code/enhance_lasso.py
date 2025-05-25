from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib


def enhance_lasso(X_train, X_test, y_train, y_test, categorical_features, numerical_features):
    print("\n=== Enhancing Lasso Regression ===")
    #dsdsdsd
    # Apply log1p transformation to target to reduce skew
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # ColumnTransformer with KNN for numeric and OrdinalEncoder for categorical
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', KNNImputer(n_neighbors=5)),
                ('scaler', StandardScaler())
            ]), numerical_features),

            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder())
            ]), categorical_features)
        ]
    )

    # Step 1: Fit preprocessor and transform data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Step 2: Feature selection using LassoCV
    selector = SelectFromModel(LassoCV(cv=5, alphas=np.logspace(-4, 1, 50), random_state=42))
    selector.fit(X_train_processed, y_train_log)
    X_train_selected = selector.transform(X_train_processed)
    X_test_selected = selector.transform(X_test_processed)

    # Step 3: Train final Lasso model on selected features
    final_model = LassoCV(cv=5, alphas=np.logspace(-4, 1, 100), random_state=42)
    final_model.fit(X_train_selected, y_train_log)

    # Step 4: Predict and reverse transformation
    y_pred_log = final_model.predict(X_test_selected)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test_log)

    # Step 5: Evaluation
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")

    # Save final pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('model', final_model)
    ])

    joblib.dump(pipeline, 'results/enhanced_lasso.joblib')
    print("Enhanced Lasso model saved to results/enhanced_lasso.joblib")

    return pipeline
