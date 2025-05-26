import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

def Demo():
    # === Step 1: Load dataset and model ===
    DATA_PATH = "../Data/Processed/enhanced_covid_data2.csv"
    MODEL_PATH = "trained_elasticnet_model.joblib"

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Define target and drop ID/date columns
    target = 'cum_positive_cases'
    X = df.drop(columns=[target, 'date', 'dates']) if 'dates' in df.columns else df.drop(columns=[target, 'date'])

    # Define demographic & socioeconomic features
    demo_socio_features = [
        'population', 'Male literacy rate %', 'Female literacy rate %',
        'Average literacy rate %', 'Female to Male ratio', 'density',
        'GDP', 'per capita in', 'Hindu', 'Muslim', 'Christian',
        'Sikhs', 'Buddhist', 'Others'
    ]
    demo_socio_features = [f for f in demo_socio_features if f in X.columns]

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # === Step 2: Extract coefficients ===
    # Try to extract from pipeline
    if hasattr(model, 'named_steps'):
        regressor = model.named_steps['regressor']
        preprocessor = model.named_steps['preprocessor']
    else:
        regressor = model
        preprocessor = None

    # Get feature names
    if preprocessor:
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = list(X.columns)
    else:
        feature_names = list(X.columns)

    # Ensure lengths match
    if len(feature_names) != len(regressor.coef_):
        print(f"❌ Mismatch: {len(regressor.coef_)} coefficients vs {len(feature_names)} feature names")
        exit()

    # Build DataFrame of all coefficients
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': regressor.coef_
    })
    coef_df['abs'] = coef_df['coefficient'].abs()

    # === Step 3: Filter only demographic/socioeconomic features ===
    demo_coef_df = coef_df[coef_df['feature'].isin(demo_socio_features)].sort_values('abs', ascending=False)

    # === Step 4: Plot ===
    plt.figure(figsize=(10, 6))
    plt.barh(demo_coef_df['feature'], demo_coef_df['coefficient'], color='skyblue')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel('Coefficient Value')
    plt.title('Demographic & Socioeconomic Feature Influence')
    plt.tight_layout()
    plt.show()

    # === Step 5: Print results ===
    print("\n📊 Demographic & Socioeconomic Feature Coefficients:")
    print(demo_coef_df[['feature', 'coefficient']])