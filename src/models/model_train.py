import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor  # Correct import for XGBRegressor
from statsmodels.tsa.arima.model import ARIMA  # Updated import for ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

def preprocess_for_xgb(df):
    df_encoded = df.copy()
    
    # Label encoding for categorical features
    label_encoders = {}
    for column in ['store', 'item_dept', 'profile', 'size']:
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column])
        label_encoders[column] = le
    
    # Drop unnecessary columns
    df_encoded = df_encoded.drop(columns=['date_id'])
    
    return df_encoded, label_encoders

def preprocess_for_lstm(df):
    # Ensure all columns are numeric
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Handle any missing values (optional: you could also drop or fill missing values)
    numeric_df = numeric_df.fillna(0)
    
    # Normalize the features (optional, but recommended for neural networks)
    X = numeric_df.drop(columns=['net_sales']).values.astype('float32')
    y = numeric_df['net_sales'].values.astype('float32')
    
    # Reshape input for LSTM [samples, timesteps, features]
    X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
    
    return X_reshaped, y

def train_arima(df, store, item_dept):
    # Filter the data for the specific store and item_dept
    series = df[(df['store'] == store) & (df['item_dept'] == item_dept)]['net_sales']
    
    # Train ARIMA model
    arima_model = ARIMA(series, order=(5, 1, 0))
    arima_fit = arima_model.fit()
    
    return arima_fit

def train_models(df):
    # Preprocess data for XGBoost
    df_xgb, label_encoders = preprocess_for_xgb(df)
    
    # Split data into training and validation sets for XGBoost
    X = df_xgb.drop(columns=['net_sales'])
    y = df_xgb['net_sales']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost model
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train, y_train)
    
    # Evaluate XGBoost
    xgb_predictions = xgb_model.predict(X_val)
    xgb_rmse = mean_squared_error(y_val, xgb_predictions, squared=False)
    print(f"XGBoost RMSE: {xgb_rmse}")
    
    # Train ARIMA models for each store and item_dept
    arima_models = {}
    for store in df['store'].unique():
        for item_dept in df['item_dept'].unique():
            arima_fit = train_arima(df, store, item_dept)
            arima_models[(store, item_dept)] = arima_fit
    
    # Preprocess data for LSTM
    X_lstm, y_lstm = preprocess_for_lstm(df)
    
    # Split data into training and validation sets for LSTM
    X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)
    
    # Train LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, verbose=1)
    
    # Evaluate LSTM
    lstm_predictions = lstm_model.predict(X_val_lstm)
    lstm_rmse = mean_squared_error(y_val_lstm, lstm_predictions, squared=False)
    print(f"LSTM RMSE: {lstm_rmse}")
    
    return xgb_model, arima_models, lstm_model, label_encoders

def evaluate_models_with_mape(df, xgb_model, arima_models, lstm_model, label_encoders):
    def calculate_mape(y_true, y_pred):
        """Calculate Mean Absolute Percentage Error (MAPE)."""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def evaluate_model(df, model, model_name, granularity):
        mape_values = []
        
        if granularity == 'date_id | store | item_dept':
            grouped = df.groupby(['store', 'item_dept', 'date_id'])
        elif granularity == 'date_id | store':
            grouped = df.groupby(['store', 'date_id'])
        else:
            raise ValueError("Unsupported granularity")
        
        for name, group in grouped:
            actual = group['net_sales']
            
            if model_name == 'XGBoost':
                # Ensure all features used in training are included
                group_encoded = group.copy()
                
                # Encode categorical features using the label encoders
                group_encoded['store'] = label_encoders['store'].transform(group_encoded['store'])
                group_encoded['item_dept'] = label_encoders['item_dept'].transform(group_encoded['item_dept'])
                group_encoded['profile'] = label_encoders['profile'].transform(group_encoded['profile'])
                group_encoded['size'] = label_encoders['size'].transform(group_encoded['size'])

                # Drop target and unnecessary columns
                group_encoded = group_encoded.drop(columns=['net_sales', 'date_id'])
                pred = model.predict(group_encoded)
                
            elif model_name == 'LSTM':
                X_lstm, _ = preprocess_for_lstm(group)
                pred = model.predict(X_lstm).flatten()
                
            elif model_name == 'ARIMA':
                pred = model[(group['store'].iloc[0], group['item_dept'].iloc[0])].forecast(steps=len(group))
            
            mape = calculate_mape(actual, pred)
            mape_values.append(mape)
        
        avg_mape = np.mean(mape_values)
        print(f"{model_name} MAPE ({granularity}): {avg_mape}%")
    
    # Evaluate XGBoost
    evaluate_model(df, xgb_model, 'XGBoost', 'date_id | store | item_dept')
    evaluate_model(df, xgb_model, 'XGBoost', 'date_id | store')
    
    # Evaluate ARIMA
    evaluate_model(df, arima_models, 'ARIMA', 'date_id | store | item_dept')
    evaluate_model(df, arima_models, 'ARIMA', 'date_id | store')
    
    # Evaluate LSTM
    evaluate_model(df, lstm_model, 'LSTM', 'date_id | store | item_dept')
    evaluate_model(df, lstm_model, 'LSTM', 'date_id | store')


grouped_df = pd.read_csv(r'D:\Lumini\NIB7072\Q5_sales_forecasting\Q5_sales_forecasting\data\output\train.csv')

xgb_model, arima_models, lstm_model, label_encoders = train_models(grouped_df)

evaluate_models_with_mape(grouped_df, xgb_model, arima_models, lstm_model, label_encoders)