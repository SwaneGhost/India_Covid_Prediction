import shap
import pandas as pd
import matplotlib.pyplot as plt

def run_shap_analysis(model, X_train, feature_names=None, max_display=9):
    """
    Run SHAP analysis on a trained ElasticNet model.

    Parameters:
        model: Trained model pipeline (with 'preprocessor' and 'regressor')
        X_train: The raw training data (before preprocessing)
        feature_names: Optional list of feature names (used if preprocessor lacks support)
        max_display: Number of top features to display in summary plot
    """
    print("🔍 Running SHAP analysis for model interpretability...")
    print("=" * 60)

    try:
        # Apply preprocessing
        preprocessor = model.named_steps['preprocessor']
        regressor = model.named_steps['regressor']

        # Transform data
        X_transformed = preprocessor.transform(X_train)

        # Use feature names if available
        if hasattr(preprocessor, 'get_feature_names_out'):
            shap_feature_names = preprocessor.get_feature_names_out()
        elif feature_names:
            shap_feature_names = feature_names
        else:
            shap_feature_names = [f'feature_{i}' for i in range(X_transformed.shape[1])]

        # Create SHAP explainer and values
        explainer = shap.Explainer(regressor.predict, X_transformed)
        shap_values = explainer(X_transformed)

        # Summary bar plot (Top N)
        print(f"\n📊 Top {max_display} Most Influential Features:")
        shap.plots.bar(shap_values, max_display=max_display, show=True)

        # Optional: Show beeswarm plot
        print(f"\n🌈 SHAP Beeswarm plot (Top {max_display} features):")
        shap.plots.beeswarm(shap_values, max_display=max_display, show=True)

        # Optional: Return SHAP values and explainer
        return shap_values, explainer

    except Exception as e:
        print(f"❌ SHAP analysis failed: {e}")
        return None, None