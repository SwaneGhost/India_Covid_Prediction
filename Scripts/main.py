from merge_data import merge_data
from clean_data import clean_data
from train import train_eval
from predict import predict

if __name__ == "__main__":
    merged_data = merge_data()
    cleaned_data = clean_data(merged_data)
    train_eval() # this will take about 1 hour to run
    predict()