from merge_data import merge_data
from clean_data import clean_data
from HGB.train import train_eval as train_eval_hgb
from HGB.predict import predict as predict_hgb
from XGB.train import train_eval as train_eval_xgb
from XGB.predict import predict as predict_xgb
from SVR.train import train_eval as train_eval_svr
from SVR.predict import predict as predict_svr
from Lasso.main import main as train_eval_lasso

if __name__ == "__main__":
    merged_data = merge_data()
    cleaned_data = clean_data(merged_data)
    
    model = "HGB"  # Specify the model to use
    
    if model == "HGB":
        train_eval_hgb() # this will take about 1 hour to run for 500 trials
        predict_hgb()
    elif model == "XGB":
        train_eval_xgb()
        predict_xgb()
    elif model == "SVR":
        train_eval_svr()
        predict_svr()
    elif model == "Lasso":
        train_eval_lasso()
    else:
        raise ValueError(f"Model {model} is not supported. Choose from 'HGB', 'XGB', or 'SVR'.")
        