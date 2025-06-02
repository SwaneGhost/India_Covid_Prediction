# Code/pipeline_utils.py

import joblib
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold

from .feature_engineering import feature_engineering, remove_highly_correlated_features
from .feature_selection import select_top_k_features


def split_features_target(df, config, drop_cols=['state', 'date', 'dates']):
    """
    Splits the dataset into features (X) and target (y), dropping metadata columns if needed.

    Parameters:
        df (DataFrame): Input dataframe
        target_col (str): Name of the target column
        drop_cols (list): Metadata or identifier columns to drop from X

    Returns:
        X (DataFrame): Features only
        y (Series): Target variable
    """
    y = df[config["target_col"]]
    X = df.drop(columns=[config["target_col"]] + [col for col in drop_cols if col in df.columns], errors='ignore')
    return X, y


def prepare_training_data(df,config):
    """
    Prepares the data for model training.

    This includes:
    - Applying feature engineering
    - Removing highly correlated features
    - Selecting top k features using f_regression
    - Restoring key columns (target, date, state)

    Parameters:
        df (DataFrame): Raw input data
        target_col (str): Name of the target column
        k (int or 'all'): Number of top features to select

    Returns:
        X_selected (DataFrame): Final selected features
        y (Series): Target values
        df_selected (DataFrame): Full frame with selected features + target + metadata
    """

    # Step 1: Feature engineering
    df = feature_engineering(df)

    # Step 2: Remove correlated columns
    df = remove_highly_correlated_features(df, config["target_col"],config["target_corr_threshold"],config["feature_corr_threshold"])

    # Step 3: Separate input features and target
    X = df.drop(columns=[config["target_col"]])
    y = df[config["target_col"]]

    # Step 4: Select top k features
    X_selected, selected_features = select_top_k_features(X, y, config)

    # Step 5: Add back non-numeric (categorical) feature and identifiers
    X_selected['state'] = df['state']
    df_selected = X_selected.copy()
    df_selected[config["target_col"]] = y

    return X_selected, y, df_selected


def print_selected_features(model_pipeline, all_feature_names):
    """
    Prints the names of numerical features selected inside the pipeline.

    It works by:
    - Accessing the preprocessor step
    - Extracting selected features after VarianceThreshold and SelectKBest

    Parameters:
        model_pipeline (Pipeline): Trained pipeline object
        all_feature_names (list): Names of all numerical features before selection
    """

    try:
        preprocessor = model_pipeline.named_steps['preprocessor']

        # Look for the 'num' transformer inside the preprocessor
        if hasattr(preprocessor, 'named_transformers_'):
            num_transformer = preprocessor.named_transformers_.get('num')

            if num_transformer is not None:
                # Get features that passed variance threshold
                if hasattr(num_transformer, 'named_steps') and 'variance_threshold' in num_transformer.named_steps:
                    vt_support = num_transformer.named_steps['variance_threshold'].get_support()
                    filtered_features = np.array(all_feature_names)[vt_support]

                    # Get final selection from SelectKBest
                    if 'select_k_best' in num_transformer.named_steps:
                        kbest_support = num_transformer.named_steps['select_k_best'].get_support()
                        final_features = filtered_features[kbest_support]

                        print("Selected numerical features:", list(final_features))
                        print(f"Total selected features: {len(final_features)}")
                    else:
                        print("Selected numerical features (after variance threshold):", list(filtered_features))
                else:
                    print("Variance threshold step not found in pipeline")
            else:
                print("Numerical transformer not found")
        else:
            print("Named transformers not available")

    except Exception as e:
        print(f"Could not extract feature names: {e}")


def get_feature_names_from_pipeline(model_pipeline):
    """
    Tries to extract final feature names from the pipeline after all preprocessing steps.

    Parameters:
        model_pipeline (Pipeline): Trained pipeline object

    Returns:
        list or None: List of feature names if available, else None
    """

    try:
        preprocessor = model_pipeline.named_steps['preprocessor']

        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names = preprocessor.get_feature_names_out()
            return feature_names
        else:
            print("Cannot extract feature names - preprocessor doesn't support get_feature_names_out")
            return None

    except Exception as e:
        print(f"Error extracting feature names: {e}")
        return None


def save_model(model, filename):
    """
    Saves the trained model or pipeline to disk using joblib.
    """
    joblib.dump(model, filename)
    print(f"\nModel saved to {filename}")


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def build_preprocessing_pipeline(numerical_features, categorical_features, use_feature_selection=False):
    """
    Builds a preprocessing pipeline for both numeric and categorical features.

    If use_feature_selection=True, adds VarianceThreshold and SelectKBest to numerical pipeline.
    """
    steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ]

    if use_feature_selection:
        steps += [
            ('variance_threshold', VarianceThreshold(threshold=0.01)),
            ('select_k_best', SelectKBest(score_func=f_regression, k='all'))
        ]

    numeric_transformer = Pipeline(steps=steps)

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor



from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(model, X_train, y_train, X_test, y_test, name="Model"):
    """
    Evaluates a trained model and prints key metrics on both train and test sets.

    Parameters:
        model: Trained pipeline or regressor
        X_train, y_train: Training data
        X_test, y_test: Test data
        name (str): Optional label for the model
    """
    print(f"\n{name} Performance Summary")
    print("=" * 50)

    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Train metrics
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = root_mean_squared_error(y_train, y_train_pred)

    # Test metrics
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    print("Training Metrics:")
    print(f"  R²: {r2_train:.4f}")
    print(f"  RMSE: {rmse_train:,.2f}")

    print("\nTest Metrics:")
    print(f"  R²: {r2_test:.4f}")
    print(f"  RMSE: {rmse_test:,.2f}")
    print(f"  MAE: {mae_test:,.2f}")

    print("\nOverfitting Check:")
    print(f"  R² Gap (Train - Test): {r2_train - r2_test:.4f}")

    print("\nPrediction Range:")
    print(f"  Predicted: {y_test_pred.min():,.0f} to {y_test_pred.max():,.0f}")
    print(f"  Actual:    {y_test.min():,.0f} to {y_test.max():,.0f}")
    print("=" * 50)
