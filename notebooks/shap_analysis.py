import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def run_shap_analysis(model, X_raw, demo_features=None, max_display=15):
    """
    Runs SHAP analysis on a trained pipeline model to interpret feature importance.

    Parameters:
        model: Trained sklearn pipeline with a 'preprocessor' and 'regressor'
        X_raw (DataFrame): Input data before transformation
        demo_features (list): List of keywords for demographic/socioeconomic features
        max_display (int): Number of top features to display in summary plot

    Returns:
        shap_values: SHAP value object
        explainer: SHAP explainer object
    """
    print("Running SHAP analysis")
    print("=" * 60)

    try:
        # Get preprocessing and model components from the pipeline
        preprocessor = model.named_steps['preprocessor']
        regressor = model.named_steps['regressor']

        # Apply preprocessing to raw input
        X_processed = preprocessor.transform(X_raw)

        # Try to get feature names from the preprocessor
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = [f'feature_{i}' for i in range(X_processed.shape[1])]

        print(f"Number of features after preprocessing: {X_processed.shape[1]}")
        print(f"Sample size for SHAP: {X_processed.shape[0]}")

        # Use a sample of the data if it's too large
        sample_size = min(1000, X_processed.shape[0])
        if X_processed.shape[0] > sample_size:
            sample_indices = np.random.choice(X_processed.shape[0], sample_size, replace=False)
            X_sample = X_processed[sample_indices]
            print(f"Using random sample of {sample_size} observations for SHAP analysis")
        else:
            X_sample = X_processed

        # Create a SHAP explainer based on model prediction
        explainer = shap.Explainer(regressor.predict, X_sample)
        shap_values = explainer(X_sample)

        # Print top features by SHAP value
        print(f"\nTop {max_display} features by SHAP value:")
        try:
            shap.plots.bar(shap_values, max_display=max_display, show=True)
        except Exception as e:
            print(f"Could not create SHAP bar plot: {e}")

        # Compute and print feature importance summary
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': np.abs(shap_values.values).mean(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)

        print("\nTop 15 most important features:")
        print(feature_importance.head(15).to_string(index=False))

        # If a list of demographic/socioeconomic features is provided, filter and report them
        if demo_features:
            demo_pattern = '|'.join([feat.lower().replace(' ', '').replace('%', '') for feat in demo_features])
            feature_names_clean = [name.lower().replace(' ', '').replace('%', '') for name in feature_importance['feature']]

            demo_mask = [
                any(demo_word in feature_names_clean[i] for demo_word in demo_pattern.split('|'))
                for i in range(len(feature_names_clean))
            ]

            df_demo = feature_importance[demo_mask].copy()

            if len(df_demo) > 0:
                print(f"\nSHAP values for {len(df_demo)} demographic/socioeconomic features:")
                print(df_demo.to_string(index=False))

                # Plot demographic SHAP summary
                plt.figure(figsize=(12, max(6, 0.4 * len(df_demo))))
                plt.barh(range(len(df_demo)), df_demo['mean_abs_shap'], color='steelblue')
                plt.yticks(range(len(df_demo)), df_demo['feature'])
                plt.title('Mean SHAP Value - Demographic and Socioeconomic Features')
                plt.xlabel('Mean Absolute SHAP Value')
                plt.tight_layout()
                plt.show()
            else:
                print("No demographic/socioeconomic features found in the selected features.")

        return shap_values, explainer

    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None
