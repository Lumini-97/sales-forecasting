import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
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

def preprocess_for_xgb(df):
    df_encoded = df.copy()
    
    label_encoders = {}
    for column in ['store', 'item_dept', 'profile', 'size']:
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column])
        label_encoders[column] = le
    
    return df_encoded, label_encoders

def bayesian_hyperparameter_tuning(train_df, test_df):
    concat_df = pd.concat([train_df, test_df])
    config = load_config()
    
    df_xgb, _ = preprocess_for_xgb(concat_df)
    
    Train = df_xgb.loc[df_xgb['date_id'] < config['data_cleaning']['train_date_id']]
    
    X_train = Train.drop(columns=[target_column, 'date_id'])
    y_train = Train[target_column]
    
    # Define the hyperparameter search space
    search_spaces = {
        'n_estimators': Integer(100, 1000),
        'max_depth': Integer(3, 10),
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0),
        'min_child_weight': Integer(1, 10),
        'gamma': Real(0, 5)
    }
    
    # Initialize the XGBoost regressor
    xgb = XGBRegressor(random_state=42)
    
    # Set up BayesSearchCV
    bayes_search = BayesSearchCV(
        estimator=xgb,
        search_spaces=search_spaces,
        n_iter=50,  # Number of parameter settings that are sampled
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    # Fit BayesSearchCV
    bayes_search.fit(X_train, y_train)
    
    # Get the best parameters
    best_params = bayes_search.best_params_
    print("Best parameters found:")
    print(best_params)
    
    # Train the model with the best parameters
    best_model = XGBRegressor(**best_params, random_state=42)
    best_model.fit(X_train, y_train)
    
    return best_params, best_model

if __name__ == "__main__":
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    best_params, best_model = bayesian_hyperparameter_tuning(train_df, test_df)
    
    # You can save the best_params to a file or print them here
    print("Best Hyperparameters:")
    print(best_params)