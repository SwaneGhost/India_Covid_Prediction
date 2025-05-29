import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def run_shap_analysis(model, X_raw, demo_features=None, max_display=15):
    print("🔍 Running Enhanced SHAP analysis for model interpretability...")
    print("=" * 60)

    try:
        preprocessor = model.named_steps['preprocessor']
        selector = model.named_steps['feature_selection']
        regressor = model.named_steps['regressor']

        # === Step 1: Preprocess raw data ===
        X_preprocessed = preprocessor.transform(X_raw)

        # === Step 2: Apply feature selection ===
        X_selected = selector.transform(X_preprocessed)

        # Get actual feature names
        all_feature_names = preprocessor.get_feature_names_out()
        selected_mask = selector.get_support()
        selected_feature_names = all_feature_names[selected_mask]

        # === Step 3: SHAP explainer ===
        explainer = shap.Explainer(regressor.predict, X_selected)
        shap_values = explainer(X_selected)

        # === Step 4: Global top features ===
        print(f"\n📊 Top {max_display} Features by SHAP Value (Overall):")
        shap.plots.bar(shap_values, max_display=max_display, show=True)

        # === Step 5: Focus on Demographic/Socioeconomic ===
        if demo_features:
            # Build SHAP summary table
            df_summary = pd.DataFrame({
                'feature': selected_feature_names,
                'mean_abs_shap': np.abs(shap_values.values).mean(axis=0)
            })

            # Match all socioeconomic/demographic features, even with 0 impact
            demo_mask = df_summary['feature'].str.contains('|'.join(demo_features))
            df_demo = df_summary[demo_mask].copy()

            # Sort for clarity (optional)
            df_demo = df_demo.sort_values('mean_abs_shap', ascending=False)

            # Print full list
            print("\n🏛 SHAP Contributions of ALL Socioeconomic & Demographic Features:")
            pd.set_option("display.max_rows", None)  # Ensure full list is shown
            print(df_demo.to_string(index=False))

            # Plot all
            plt.figure(figsize=(10, max(6, 0.4 * len(df_demo))))
            plt.barh(df_demo['feature'], df_demo['mean_abs_shap'], color='cornflowerblue')
            plt.title('SHAP Impact: All Socioeconomic & Demographic Features')
            plt.xlabel('Mean |SHAP Value|')
            plt.tight_layout()
            plt.show()

        return shap_values, explainer

    except Exception as e:
        print(f"❌ SHAP analysis failed: {e}")
        return None, None
