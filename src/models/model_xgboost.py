import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
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
xgb_model_path = config['model_paths']['xgboost']
xgb_encoders_path = config['model_paths']['xgboost_encoders']

def preprocess_for_xgb(df):
    df_encoded = df.copy()
    
    label_encoders = {}
    for column in ['store', 'item_dept', 'profile', 'size']:
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column])
        label_encoders[column] = le
    
    # df_encoded = df_encoded.drop(columns=['date_id'])
    
    return df_encoded, label_encoders

def train_xgb_model(train_df,test_df):
    concat_df = pd.concat([train_df, test_df])
    config = load_config()
    
    df_xgb, label_encoders = preprocess_for_xgb(concat_df)
    
    # X = df_xgb.drop(columns=[target_column])
    # y = df_xgb[target_column]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_train = X.loc[X['date_id']<config['data_cleaning']['train_date_id']]
    # X_test = X.loc[X['date_id']>=config['data_cleaning']['train_date_id']]
    # y_train = y.loc[y['date_id']<config['data_cleaning']['train_date_id']]
    # y_test = y.loc[y['date_id']>=config['data_cleaning']['train_date_id']]
    

    Train = df_xgb.loc[df_xgb['date_id']<config['data_cleaning']['train_date_id']]
    Test = df_xgb.loc[df_xgb['date_id']>=config['data_cleaning']['train_date_id']]
    
    X_train = Train.drop(columns=[target_column])
    y_train = Train[target_column]
    X_test = Test.drop(columns=[target_column])
    y_test = Test[target_column]

    X_train = X_train.drop(columns=['date_id'])
    X_test = X_test.drop(columns=['date_id'])
    # y_train = y_train.drop(columns=['date_id'])
    # y_test = y_test.drop(columns=['date_id'])
    
    xgb_model = XGBRegressor(**config['models']['xgboost'])
    xgb_model.fit(X_train, y_train)
    
    xgb_predictions = xgb_model.predict(X_test)
    xgb_rmse = mean_squared_error(y_test, xgb_predictions, squared=False)
    print(f"XGBoost RMSE: {xgb_rmse}")
    
    # Save the model and label encoders as pkl files
    with open(config['model_paths']['xgboost'], 'wb') as f:
        pickle.dump(xgb_model, f)
    with open(config['model_paths']['xgboost_encoders'], 'wb') as f:
        pickle.dump(label_encoders, f)
    
    return xgb_model, label_encoders

def load_models_and_encoders():
    models = {}
    for name, path in [('XGB Model', xgb_model_path)]:
        model = load_pickle(path)
        if model is not None:
            models[name] = model
    
    xgb_encoders = load_pickle(xgb_encoders_path)
    return models, xgb_encoders

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

if __name__ == "__main__":
    # df = pd.read_csv(train_data_path)
    # df = df.drop(columns=['date_id'])
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    train_xgb_model(train_df,test_df)