def Demo(model, df):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    print("📊 Analyzing Demographic & Socioeconomic Feature Contributions...")

    # List your known demographic/socioeconomic features:
    demo_socio_features = [
        'population', 'area', 'GDP',
        'Hindu', 'Muslim', 'Christian', 'Sikhs', 'Buddhist', 'Others',
        'primary_health_centers', 'community_health_centers', 'sub_district_hospitals',
        'district_hospitals', 'public_health_facilities', 'public_beds',
        'rural_hospitals', 'rural_beds', 'urban_hospitals', 'urban_beds',
        'total_beds', 'beds_per_1000'
    ]

    # Drop target and identifiers if present
    drop_cols = ['cum_positive_cases', 'date', 'dates']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df['cum_positive_cases']

    # Match only features that exist in both the model and dataset
    model_features = X.columns.tolist()
    demo_features_used = [f for f in demo_socio_features if f in model_features]

    if hasattr(model.named_steps['regressor'], 'coef_'):
        coefs = model.named_steps['regressor'].coef_
        feature_names = X.columns

        if len(coefs) != len(feature_names):
            print(f"❌ Mismatch: {len(coefs)} coefficients vs {len(feature_names)} feature names")
            return

        # Build DataFrame of coefficients
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefs,
            'AbsCoefficient': np.abs(coefs)
        })

        # Filter only the demographic and socioeconomic features
        filtered_df = coef_df[coef_df['Feature'].isin(demo_features_used)].sort_values('AbsCoefficient', ascending=False)

        print("\n🏛 Top Demographic & Socioeconomic Predictors:")
        print(filtered_df[['Feature', 'Coefficient']].to_string(index=False))

        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(filtered_df['Feature'], filtered_df['Coefficient'], color='skyblue')
        plt.title('Demographic & Socioeconomic Feature Impact on Predictions')
        plt.axvline(0, color='red', linestyle='--')
        plt.tight_layout()
        plt.show()

    else:
        print("❌ Model does not expose coefficients (not a linear model).")