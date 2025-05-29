import pandas as pd
import os
import matplotlib.pyplot as plt
import yaml

from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold

def predict():
    """
    Plot the predictions on a gt vs predictions plot using the median hyperparameters
    across all folds.
    """

    # load config file
    config_path = os.path.join("Configs", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    last_run_folder = os.path.join(config["output_dir"], config["last_run_folder"])
    
    # read the median hyperparameters from the YAML file
    median_params_path = os.path.join(last_run_folder, "median_params.yaml")
    
    if not os.path.exists(median_params_path):
        raise FileNotFoundError(f"Median parameters file not found at {median_params_path}")
    
    with open(median_params_path, 'r') as file:
        median_params = yaml.safe_load(file)

    # Load the data
    df = pd.read_csv(config["data_path"])

    # Reset index after filtering
    df.reset_index(drop=True, inplace=True)

    # Prepare the features and target variable
    X = df.drop(columns=["target", "state", "dates"])
    y = df["target"]
    groups = df["state"]

    # Load the model
    model = XGBRegressor(**median_params)

    # Save ground truth and predictions
    gt_and_preds = pd.DataFrame(columns=["ground_truth", "predictions"])
    gt_and_preds["ground_truth"] = y

    # Cross-predict using the model
    outer_cv = GroupKFold(n_splits=5)

    for train_idx, test_idx in outer_cv.split(X, y, groups):
        # Split the data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, _ = y.iloc[train_idx], y.iloc[test_idx]

        # Fit the model
        model.fit(X_train, y_train)

        # Predict on the test set
        preds = model.predict(X_test)

        # Store predictions
        gt_and_preds.loc[test_idx, "predictions"] = preds

    # Save the predictions to a CSV file
    gt_and_preds.to_csv(os.path.join("Runs", "run_20250526_204409", "gt_vs_predictions.csv"), index=False)

    # Plot and save the predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(gt_and_preds["ground_truth"], gt_and_preds["predictions"], alpha=0.5)
    plt.plot([gt_and_preds["ground_truth"].min(), gt_and_preds["ground_truth"].max()],
             [gt_and_preds["ground_truth"].min(), gt_and_preds["ground_truth"].max()],
             color='red', linestyle='--')
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.title("Ground Truth vs Predictions")
    plt.savefig(os.path.join("Figures", "gt_vs_predictions_plot_XGBRegressor.png"))
    plt.show()
