import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def run_shap_analysis(model, X_raw, demo_features=None, max_display=15):
    """
    Run SHAP analysis to interpret model predictions.
    Fixed to handle the new pipeline structure correctly.

    Parameters:
        model: Trained sklearn pipeline with steps: preprocessor, regressor
        X_raw (pd.DataFrame): Raw input features before transformation
        demo_features (list): List of demographic/socioeconomic feature keywords to track
        max_display (int): Number of top features to display in global SHAP summary

    Returns:
        shap_values, explainer
    """
    print("Running SHAP analysis...")
    print("=" * 60)

    try:
        # Get the preprocessor and regressor from the pipeline
        preprocessor = model.named_steps['preprocessor']
        regressor = model.named_steps['regressor']

        # Apply preprocessing to get the final feature matrix
        X_processed = preprocessor.transform(X_raw)

        # Get feature names after preprocessing
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            # Fallback if get_feature_names_out is not available
            feature_names = [f'feature_{i}' for i in range(X_processed.shape[1])]

        print(f"Number of features after preprocessing: {X_processed.shape[1]}")
        print(f"Sample size for SHAP: {X_processed.shape[0]}")

        # Create SHAP explainer for the regressor only (using preprocessed data)
        # Use a sample of data if dataset is too large
        sample_size = min(1000, X_processed.shape[0])
        if X_processed.shape[0] > sample_size:
            sample_indices = np.random.choice(X_processed.shape[0], sample_size, replace=False)
            X_sample = X_processed[sample_indices]
            print(f"Using random sample of {sample_size} observations for SHAP analysis")
        else:
            X_sample = X_processed

        # Create explainer
        explainer = shap.Explainer(regressor.predict, X_sample)
        shap_values = explainer(X_sample)

        # Global feature importance plot
        print(f"\nTop {max_display} features by SHAP value:")
        try:
            shap.plots.bar(shap_values, max_display=max_display, show=True)
        except Exception as e:
            print(f"Could not create SHAP bar plot: {e}")

        # Summary of feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': np.abs(shap_values.values).mean(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)

        print("\nTop 15 most important features:")
        print(feature_importance.head(15).to_string(index=False))

        # Focus on demographic/socioeconomic features if specified
        if demo_features:
            # Create a more flexible matching pattern
            demo_pattern = '|'.join([feat.lower().replace(' ', '').replace('%', '')
                                     for feat in demo_features])

            # Match features (case-insensitive, flexible matching)
            feature_names_clean = [name.lower().replace(' ', '').replace('%', '')
                                   for name in feature_importance['feature']]

            demo_mask = [any(demo_word in feat_name for demo_word in demo_pattern.split('|')
                             for feat_name in [feature_names_clean[i]])
                         for i in range(len(feature_names_clean))]

            df_demo = feature_importance[demo_mask].copy()

            if len(df_demo) > 0:
                print(f"\nSHAP values for {len(df_demo)} demographic/socioeconomic features:")
                print(df_demo.to_string(index=False))

                # Plot demographic features
                plt.figure(figsize=(12, max(6, 0.4 * len(df_demo))))
                plt.barh(range(len(df_demo)), df_demo['mean_abs_shap'], color='steelblue')
                plt.yticks(range(len(df_demo)), df_demo['feature'])
                plt.title('Mean SHAP Value - Demographic and Socioeconomic Features')
                plt.xlabel('Mean Absolute SHAP Value')
                plt.tight_layout()
                plt.show()
            else:
                print("\nNo demographic/socioeconomic features found in the selected features.")

        return shap_values, explainer

    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None