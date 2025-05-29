import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def run_shap_analysis(model, X_raw, demo_features=None, max_display=15):
    """
    Run SHAP analysis to interpret model predictions.

    Parameters:
        model: Trained sklearn pipeline with steps: preprocessor, feature_selection, regressor
        X_raw (pd.DataFrame): Raw input features before transformation
        demo_features (list): List of demographic/socioeconomic feature keywords to track
        max_display (int): Number of top features to display in global SHAP summary

    Returns:
        shap_values, explainer
    """
    print("Running SHAP analysis...")
    print("=" * 60)

    try:
        preprocessor = model.named_steps['preprocessor']
        selector = model.named_steps['feature_selection']
        regressor = model.named_steps['regressor']

        # Step 1: Apply preprocessing
        X_preprocessed = preprocessor.transform(X_raw)

        # Step 2: Feature selection
        X_selected = selector.transform(X_preprocessed)

        # Step 3: Get feature names after preprocessing and selection
        all_feature_names = preprocessor.get_feature_names_out()
        selected_mask = selector.get_support()
        selected_feature_names = all_feature_names[selected_mask]

        # Step 4: Run SHAP Explainer
        explainer = shap.Explainer(regressor.predict, X_selected)
        shap_values = explainer(X_selected)

        # Step 5: Global feature importance
        print(f"\nTop {max_display} features by SHAP value:")
        shap.plots.bar(shap_values, max_display=max_display, show=True)

        # Step 6: Focus on demographic/socioeconomic features
        if demo_features:
            df_summary = pd.DataFrame({
                'feature': selected_feature_names,
                'mean_abs_shap': np.abs(shap_values.values).mean(axis=0)
            })

            # Filter features that match any of the demographic keywords
            demo_mask = df_summary['feature'].str.contains('|'.join(demo_features))
            df_demo = df_summary[demo_mask].copy()
            df_demo = df_demo.sort_values('mean_abs_shap', ascending=False)

            # Print summary
            print("\nSHAP values for selected demographic/socioeconomic features:")
            pd.set_option("display.max_rows", None)
            print(df_demo.to_string(index=False))

            # Plot selected features
            plt.figure(figsize=(10, max(6, 0.4 * len(df_demo))))
            plt.barh(df_demo['feature'], df_demo['mean_abs_shap'], color='steelblue')
            plt.title('Mean SHAP Value - Demographic and Socioeconomic Features')
            plt.xlabel('Mean Absolute SHAP Value')
            plt.tight_layout()
            plt.show()

        return shap_values, explainer

    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        return None, None
