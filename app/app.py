import base64
import streamlit as st
import pandas as pd
import pickle
import yaml
import io
import traceback
from sklearn.preprocessing import LabelEncoder

# Load your models and encoders
def load_config(config_path='./config/config.yaml'):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        st.error(f"Error loading config: {str(e)}")
        return None

def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading file from {file_path}: {str(e)}")
        return None

config = load_config()
if config is None:
    st.error("Failed to load configuration. Please check your config file.")
    st.stop()

xgb_model_path = config['model_paths']['xgboost']
arima_model_path = config['model_paths']['arima']
lstm_model_path = config['model_paths']['lstm']
xgb_encoders_path = config['model_paths']['xgboost_encoders']
pred_output_columns = config['data']['output']['output_columns']

def load_models_and_encoders():
    models = {}
    for name, path in [('XGB Model', xgb_model_path), ('ARIMA Model', arima_model_path), ('LSTM Model', lstm_model_path)]:
        model = load_pickle(path)
        if model is not None:
            models[name] = model
    
    xgb_encoders = load_pickle(xgb_encoders_path)
    return models, xgb_encoders

def preprocess_for_xgboost(data, encoders):
    if encoders is None:
        st.error("XGBoost encoders not loaded. Cannot preprocess data.")
        return None

    try:
        # Convert date_id to datetime and then to integer (number of days since a reference date)
        # data['date_id'] = pd.to_datetime(data['date_id'])
        if 'date_id' in data.columns:
            data.drop(columns=['date_id'], inplace=True)
        if 'daily_sales_qty' in data.columns:
            data.drop(columns=['daily_sales_qty'], inplace=True)
        # reference_date = data['date_id'].min()
        # data['date_id'] = (data['date_id'] - reference_date).dt.days

        # Apply encoding to categorical variables
        for column, encoder in encoders.items():
            if column in data.columns:
                if isinstance(encoder, LabelEncoder):
                    # For LabelEncoder, we need to handle unseen categories
                    data[column] = data[column].astype(str)
                    unseen = set(data[column]) - set(encoder.classes_)
                    if unseen:
                        st.warning(f"Unseen categories in {column}: {unseen}. These will be set to the most frequent category.")
                        most_frequent = encoder.inverse_transform([encoder.transform(encoder.classes_).max()])[0]
                        data.loc[data[column].isin(unseen), column] = most_frequent
                data[column] = encoder.transform(data[column])

        # Ensure all columns are numeric
        for col in data.columns:
            if data[col].dtype == 'object':
                st.warning(f"Column {col} is still of type object. Attempting to convert to numeric.")
                data[col] = pd.to_numeric(data[col], errors='coerce')

        return data
    except Exception as e:
        st.error(f"Error during XGBoost preprocessing: {str(e)}")
        return None

# Perform inference
def predict(model, data, model_name, encoders=None):
    try:
        if model_name == 'XGB Model' and encoders is not None:
            data = preprocess_for_xgboost(data, encoders)
            if data is None:
                return None
            st.write("Preprocessed Data Types:")
            st.write(data.dtypes)
            return model.predict(data)
        elif isinstance(model, pd.DataFrame):  # ARIMA model
            return model['forecast']  # Assuming ARIMA model returns a DataFrame with 'forecast' column
        else:  # LSTM
            return model.predict(data)
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.text("Traceback:")
        st.text(traceback.format_exc())
        return None

# Main app
def main():
    st.title('Sales Forecasting App')

    # Load models and encoders
    models, xgb_encoders = load_models_and_encoders()
    if not models:
        st.error("No models were loaded successfully. Please check your model files.")
        st.stop()

    # Model selection dropdown
    selected_model = st.selectbox('Select a model', list(models.keys()))

    # File uploader for test dataset
    uploaded_file = st.file_uploader("Choose a test CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            df_clone = df.copy()
            st.write("Dataset Preview:")
            st.write(df.head())

            # Perform inference button
            if st.button('Perform Inference'):
                # Perform prediction
                model = models[selected_model]
                predictions = predict(model, df, selected_model, xgb_encoders if selected_model == 'XGB Model' else None)
                if predictions is not None:
                    df['prediction'] = predictions
                    df_clone = df_clone[pred_output_columns]
                    df_clone['prediction'] = predictions
                    st.write("Predictions added to dataset:")
                    st.write(df_clone.head())

                    # Create a download button for the resulting CSV
                    csv = df_clone.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                else:
                    st.error("Failed to generate predictions. Please check the error messages above.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.text("Traceback:")
            st.text(traceback.format_exc())

if __name__ == '__main__':
    main()