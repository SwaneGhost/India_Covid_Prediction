from sklearn.metrics import root_mean_squared_error
import yaml
import os
import pandas as pd
import numpy as np

import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold

def train_eval():
    """
    Train and evaluate the model.
    """
    # Load configuration parameters
    config = load_config()

    # Check if the output directory exists, if not create it
    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])
    
    # Create a foler of the current run with a timestamp
    run_folder = os.path.join(config["output_dir"], f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)

    # Save the config file to the run folder
    config_path = os.path.join(run_folder, "config.yaml")
    with open(config_path, "w") as file:
        yaml.dump(config, file)
    
    # Load the dataset
    df = pd.read_csv(config["data_path"])
    
    # Set seed for reproducibility
    np.random.seed(config["seed"])

    X = df.drop(columns=["target"])
    y = df["target"]
    groups = df["state"]

    outer_cv = GroupKFold(n_splits=5)
    best_params_list = []
    best_params_train = []
    best_params_test = []

    # Outer cross-validation loop
    for train_idx, test_idx in outer_cv.split(X, y, groups):

        # Outer cv split
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Get the groups for the current folds test
        groups_train = groups.iloc[train_idx]

        def objective(trial):
            # TODO - revise hyperparameter search space
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": config["seed"],
            }
            # Inner CV for hyperparameter tuning
            inner_cv = GroupKFold(n_splits=3)
            scores = []
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train, groups_train):
                # Inner cv split
                X_inner_train, X_inner_val = X_train.iloc[inner_train_idx], X_train.iloc[inner_val_idx]
                y_inner_train, y_inner_val = y_train.iloc[inner_train_idx], y_train.iloc[inner_val_idx]
                model = XGBRegressor(**params)
                model.fit(X_inner_train, y_inner_train)
                y_pred = model.predict(X_inner_val)
                score = root_mean_squared_error(y_inner_val, y_pred)
                scores.append(score)
            return np.mean(scores)  # Optuna minimizes RMSE

        # Create a study for hyperparameter optimization
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=config["seed"]))
        
        # Optimize the hyperparameters using the objective function
        study.optimize(objective, n_trials=config["n_trials"], show_progress_bar=True)
        
        # Train the model with the best hyperparameters on the full training set
        best_params = study.best_params
        model = XGBRegressor(**best_params)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate RMSE for training and test sets
        train_rmse = root_mean_squared_error(y_train, y_train_pred)
        test_rmse = root_mean_squared_error(y_test, y_test_pred)

        # Store the results
        best_params_train.append(train_rmse)
        best_params_test.append(test_rmse)
        best_params_list.append(study.best_params)

    # Find the index of the median RMSE for the test set
    median_index = np.argsort(best_params_test)[len(best_params_test) // 2]
    median_params = best_params_list[median_index]

    print("Median hyperparameters across all folds:", median_params)

    # Save the loss values and hyperparameters
    results_df = pd.DataFrame({
        "fold": range(1, len(best_params_list) + 1),
        "train_rmse": best_params_train,
        "test_rmse": best_params_test
    })
    results_df.to_csv(os.path.join(run_folder, "results.csv"), index=False)

    # Save the median hyperparameters as a YAML file
    median_params_path = os.path.join(run_folder, "median_params.yaml")
    with open(median_params_path, "w") as file:
        yaml.dump(median_params, file)

    return median_params



def load_config() -> dict:
    """
    Load parameters from the config.yaml file and check for required keys.
    Returns:
        dict: A dictionary containing the parameters.
    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If any required key is missing.
    """
    path = "Config/config.yaml"

    required_keys = ["data_path"]

    # Check if the path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found at {path}")
    
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    # Check for required keys
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required key: {key} in config.yaml")
        
    # Set default values for optional keys
    config.setdefault("output_dir", "Runs/")
    config.setdefault("seed", 42)
    config.setdefault("n_trials", 100)

    return config