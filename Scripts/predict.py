import pandas as pd
import os
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold

def predict():
    """
    Plot the predictions on a gt vs predictions plot using the median hyperparameters
    across all folds.
    """

    median_params = {
        'colsample_bytree': 0.895100074311251,
        'learning_rate': 0.2702113506729492,
        'max_depth': 2,
        'n_estimators': 131,
        'reg_alpha': 0.3059470293939967,
        'reg_lambda': 0.8402106219103826,
        'subsample': 0.9903791207778909,
    }

    # Load the data
    df = pd.read_csv(os.path.join("Data", "Train", "train_data.csv"))

    # Remove "Maharashtra" states from the dataset
    df = df[~df["state"].isin(["Maharashtra"])]

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
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

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
