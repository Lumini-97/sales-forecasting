import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
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
test_data_path = config['data']['processed']['featured_test_data_output_path']


def preprocess_for_lstm(train_df, test_df, target_column, config):
    # Concatenate train and test dataframes
    combined_df = pd.concat([train_df, test_df])
    
    # Select numeric columns and fill missing values
    numeric_df = combined_df.select_dtypes(include=[np.number])
    numeric_df = numeric_df.fillna(0)
    
    # Separate features and target
    X = numeric_df.drop(columns=[target_column]).values.astype('float32')
    y = numeric_df[target_column].values.astype('float32')
    
    # Reshape features for LSTM
    X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
    
    # Add the reshaped features and target back to the combined dataframe
    combined_df['X_reshaped'] = list(X_reshaped)
    combined_df['y'] = y
    
    # Split the combined dataframe back into train and test sets based on date_id
    train_date_id = config['data_cleaning']['train_date_id']
    train = combined_df.loc[combined_df['date_id'] < train_date_id]
    test = combined_df.loc[combined_df['date_id'] >= train_date_id]
    
    # Extract the reshaped features and target from the train and test sets
    X_train = np.array(train['X_reshaped'].tolist())
    y_train = train['y'].values
    X_test = np.array(test['X_reshaped'].tolist())
    y_test = test['y'].values
    
    return X_train, y_train, X_test, y_test

def train_lstm_model(train_df, test_df):
    config = load_config()
    
    X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm = preprocess_for_lstm(train_df, test_df, 'daily_sales_qty', config)
    
    lstm_model = Sequential()
    lstm_model.add(LSTM(config['models']['lstm']['units'], activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=config['models']['lstm']['epochs'], verbose=0)
    
    lstm_predictions = lstm_model.predict(X_val_lstm)
    lstm_rmse = mean_squared_error(y_val_lstm, lstm_predictions, squared=False)
    print(f"LSTM RMSE: {lstm_rmse}")
    
    # Save the model as a pkl file
    with open(config['model_paths']['lstm'], 'wb') as f:
        pickle.dump(lstm_model, f)
    
    return lstm_model

if __name__ == "__main__":
    # config = load_config()
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    train_lstm_model(train_df, test_df)
