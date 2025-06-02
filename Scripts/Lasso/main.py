import pandas as pd
import warnings
import os

from .HyperTuning import tune_lasso_model
from .model_train import train_lasso_model
import yaml

# Suppress convergence warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='Objective did not converge')

from .pipeline_utils import (
    prepare_training_data,
    print_selected_features,
    save_model,
    get_feature_names_from_pipeline, evaluate_model
)

def main():
    # Load configuration
    with open("/Config/lasso/configs.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Step 1: Load dataset
    df = pd.read_csv('Data/Train/train_data.csv')
    # Step 2: Feature engineering and selection
    print("\nStep 2: Feature engineering and selection")
    X_selected, y_raw, df_selected = prepare_training_data(df, config)
    y = y_raw

    # Step 3: Train lasso model
    print("\nStep 3: Training lasso model")
    try:
        model, X_train_final, X_test_final, y_train, y_test = train_lasso_model(df_selected)

        print("\nEvaluating predictions ...")
        evaluate_model(model, X_train_final, y_train, X_test_final, y_test, name="Baseline Lasso")

    except Exception as e:
        print(f"Model training failed: {e}")

    # Step 4: Hyperparameter tuning
    print("\nStep 4: Hyperparameter tuning")
    categorical_features = ['state']
    numerical_features = X_selected.select_dtypes(include=['float64', 'int64']) \
                                    .columns.difference(categorical_features).tolist()

    print(f"Categorical features: {categorical_features}")
    print(f"Number of numerical features: {len(numerical_features)}")

    best_model = tune_lasso_model(
        X_selected, y,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        n_trials=config["n_trails"]
    )

    # Step 5: Feature analysis
    print("\nStep 5: Selected Features Analysis:")
    print_selected_features(best_model, numerical_features)

    final_feature_names = get_feature_names_from_pipeline(best_model)
    if final_feature_names is not None:
        print(f"Total features in final model: {len(final_feature_names)}")

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    # Step 6: Save trained model
    save_model(best_model, f"Models/trained_lasso_model{timestamp}.joblib")

    # Step 7: Final Model Evaluation
    print("\nStep 7: Final Model Evaluation")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=config["test_size"], random_state=config["seed"])

    best_model.fit(X_train, y_train)
    evaluate_model(best_model, X_train, y_train, X_test, y_test, name="Tuned Lasso")



