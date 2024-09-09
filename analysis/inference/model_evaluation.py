import pandas as pd
import numpy as np
import yaml
import pickle
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
import xgboost as xgb

warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore")

def model_evaluation():
    def load_model(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            return model

    def load_config(config_path='config/config.yaml'):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    config = load_config()
    df = pd.read_csv(config['data']['processed']['featured_test_data_output_path'])
    xgb_model = load_model(config['model_paths']['xgboost'])
    arima_models = load_model(config['model_paths']['arima'])
    lstm_model = load_model(config['model_paths']['lstm'])
    xgb_encoders = load_model(config['model_paths']['xgboost_encoders'])
    target_column = config['target_column']

    # def preprocess_for_xgb(df):
    #     df_encoded = df.copy()
        
    #     # Label encoding for categorical features
    #     label_encoders = {}
    #     for column in ['store', 'item_dept', 'profile', 'size']:
    #         le = LabelEncoder()
    #         df_encoded[column] = le.fit_transform(df_encoded[column])
    #         label_encoders[column] = le
        
    #     # Drop unnecessary columns
    #     df_encoded = df_encoded.drop(columns=['primary_key', 'date_id'])
        
    #     return df_encoded, label_encoders

    def preprocess_for_lstm(df):
        # Ensure all columns are numeric
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Handle any missing values (optional: you could also drop or fill missing values)
        numeric_df = numeric_df.fillna(0)
        
        # Normalize the features (optional, but recommended for neural networks)
        X = numeric_df.drop(columns=[target_column]).values.astype('float32')
        y = numeric_df[target_column].values.astype('float32')
        
        # Reshape input for LSTM [samples, timesteps, features]
        X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
        
        return X_reshaped, y

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
                actual = group[target_column]
                
                if model_name == 'XGBoost':
                    # Ensure all features used in training are included
                    group_encoded = group.copy()
                    
                    # Encode categorical features using the label encoders
                    group_encoded['store'] = label_encoders['store'].transform(group_encoded['store'])
                    group_encoded['item_dept'] = label_encoders['item_dept'].transform(group_encoded['item_dept'])
                    group_encoded['profile'] = label_encoders['profile'].transform(group_encoded['profile'])
                    group_encoded['size'] = label_encoders['size'].transform(group_encoded['size'])

                    # Drop target and unnecessary columns
                    group_encoded = group_encoded.drop(columns=[target_column, 'date_id'])
                    with xgb.config_context(verbosity=0):
                        pred = model.predict(group_encoded)
                    
                elif model_name == 'LSTM':
                    X_lstm, _ = preprocess_for_lstm(group)
                    # pred = model.predict(X_lstm).flatten()
                    pred = model.predict(X_lstm, verbose=0).flatten()  # Suppress progress bar

                    
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

    evaluate_models_with_mape(df, xgb_model, arima_models, lstm_model, xgb_encoders)
# if __name__ == "__main__":
#     model_evaluation()