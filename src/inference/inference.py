import pandas as pd
import pickle
import sys
from sklearn.preprocessing import LabelEncoder
import yaml

# Helper functions
def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        print(f"Error loading file from {file_path}: {str(e)}")
        sys.exit(1)

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def preprocess_for_xgboost(data, encoders):
    if encoders is None:
        raise ValueError("XGBoost encoders not provided. Cannot preprocess data.")

    # Drop columns that are not needed
    if 'date_id' in data.columns:
        data.drop(columns=['date_id'], inplace=True)
    if 'daily_sales_qty' in data.columns:
        data.drop(columns=['daily_sales_qty'], inplace=True)

    # Apply encoding to categorical variables
    for column, encoder in encoders.items():
        if column in data.columns:
            if isinstance(encoder, LabelEncoder):
                data[column] = data[column].astype(str)
                unseen = set(data[column]) - set(encoder.classes_)
                if unseen:
                    print(f"Warning: Unseen categories in {column}: {unseen}. These will be set to the most frequent category.")
                    most_frequent = encoder.inverse_transform([encoder.transform(encoder.classes_).max()])[0]
                    data.loc[data[column].isin(unseen), column] = most_frequent
                data[column] = encoder.transform(data[column])

    # Ensure all columns are numeric
    for col in data.columns:
        if data[col].dtype == 'object':
            print(f"Warning: Column {col} is still of type object. Attempting to convert to numeric.")
            data[col] = pd.to_numeric(data[col], errors='coerce')

    return data

def predict(model, data, model_name, encoders=None):
    try:
        if model_name == 'XGB Model' and encoders is not None:
            data = preprocess_for_xgboost(data, encoders)
            if data is None:
                return None
            return model.predict(data)
        elif isinstance(model, pd.DataFrame):  # ARIMA model
            return model['forecast']  # Assuming ARIMA model returns a DataFrame with 'forecast' column
        else:  # LSTM
            return model.predict(data)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

def run_inference():
    # Paths to the model and encoder files
    config = load_config()
    predicted_output_path = config['data']['output']['output_data']
    xgb_model_path = config['model_paths']['xgboost']
    arima_model_path = config['model_paths']['arima']
    lstm_model_path = config['model_paths']['lstm']
    xgb_encoders_path = config['model_paths']['xgboost_encoders']
    selected_model = 'XGB Model'
    pred_output_columns = config['data']['output']['output_columns']
    # Load models and encoders
    xgb_model = load_pickle(xgb_model_path)
    arima_model = load_pickle(arima_model_path)
    lstm_model = load_pickle(lstm_model_path)
    xgb_encoders = load_pickle(xgb_encoders_path)

    # Load test data
    test_data_path = config['data']['processed']['featured_test_data_output_path']
    df = pd.read_csv(test_data_path)
    test_clone = df.copy()


    if selected_model == 'XGB Model':
        model = xgb_model
        encoders = xgb_encoders
    elif selected_model == 'ARIMA Model':
        model = arima_model
        encoders = None
    elif selected_model == 'LSTM Model':
        model = lstm_model
        encoders = None
    else:
        print("Error: Invalid model selection.")
        sys.exit(1)

    # Perform inference
    predictions = predict(model, df, selected_model, encoders)
    if predictions is not None:
        # df['prediction'] = predictions
        test_clone = test_clone[pred_output_columns]
        test_clone['prediction'] = predictions
        # Save the results
        test_clone.to_csv(predicted_output_path, index=False)
        print("Predictions saved to 'predictions.csv'.")
    else:
        print("Failed to generate predictions.")

if __name__ == '__main__':
    run_inference()
