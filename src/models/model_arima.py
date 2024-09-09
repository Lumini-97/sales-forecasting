import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pickle
import yaml
import warnings
warnings.filterwarnings("ignore")



def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config()
target_column = config['target_column']
train_data_path = config['data']['processed']['featured_train_data_output_path']


def train_arima(df, store, item_dept, config):
    series = df[(df['store'] == store) & (df['item_dept'] == item_dept)][target_column]
    
    arima_model = ARIMA(series, order=tuple(config['models']['arima']['order']))
    arima_fit = arima_model.fit() 
    
    return arima_fit

def train_arima_models(df):
    config = load_config()
    
    arima_models = {}
    for store in df['store'].unique():
        for item_dept in df['item_dept'].unique():
            arima_fit = train_arima(df, store, item_dept, config)
            arima_models[(store, item_dept)] = arima_fit
    
    # Save the models as a pkl file
    with open(config['model_paths']['arima'], 'wb') as f:
        pickle.dump(arima_models, f)
    
    return arima_models

if __name__ == "__main__":
    config = load_config()
    df = pd.read_csv(train_data_path)
    df = df.drop(columns=['date_id'])
    train_arima_models(df)