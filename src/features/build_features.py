import pandas as pd
import yaml
import warnings
import holidays
from datetime import date

warnings.filterwarnings("ignore")

def build_features():
    def load_config(config_path='config/config.yaml'):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def is_holiday(date):
        uk_holidays = holidays.UnitedKingdom()
        return 1 if date in uk_holidays else 0

    def add_time_features(df):
        df['date_id'] = pd.to_datetime(df['date_id'])
        df['day_of_week'] = df['date_id'].dt.dayofweek
        df['day_of_month'] = df['date_id'].dt.day
        df['month'] = df['date_id'].dt.month
        df['year'] = df['date_id'].dt.year
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_holiday'] = df['date_id'].dt.date.map(is_holiday)
        return df

    def add_sales_features(df, config):
        window_sizes = config['feature_engineering']['rolling_window_sizes']
        lag_days = config['feature_engineering']['lag_days']

        for window in window_sizes:
            df[f'rolling_avg_{window}d'] = df.groupby(['store', 'item_dept'])['daily_sales_qty'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())

        df['cumulative_sales'] = df.groupby(['store', 'item_dept'])['daily_sales_qty'].cumsum()

        for lag in lag_days:
            df[f'lag_{lag}d'] = df.groupby(['store', 'item_dept'])['daily_sales_qty'].shift(lag)
            df[f'lag_{lag}d'].fillna(0, inplace=True)

        return df

    def add_outlet_features(df, outlet_features_path):
        outlet_features = pd.read_csv(outlet_features_path)
        return pd.merge(df, outlet_features, on='store', how='left')

    def main():
        try:
            config = load_config()
            train_data_path = config['data']['processed']['processed_train_data_path']
            test_data_path = config['data']['processed']['processed_test_data_path']
            outlet_features_path = config['data']['raw']['outlet_info_path']
            output_train_path = config['data']['processed']['featured_train_data_output_path']
            output_test_path = config['data']['processed']['featured_test_data_output_path']

            # Load and combine train and test data
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            # df = pd.concat([train_df, test_df], ignore_index=True)

            # Add features
            train_df = add_time_features(train_df)
            train_df = add_sales_features(train_df, config)
            train_df = add_outlet_features(train_df, outlet_features_path)
            # train_df = train_df.drop(['date_id'], axis=1)
            test_df = add_time_features(test_df)
            test_df = add_sales_features(test_df, config)
            test_df = add_outlet_features(test_df, outlet_features_path)    
            # test_df = test_df.drop(['date_id'], axis=1)

            # Save the featured dataset
            train_df.to_csv(output_train_path, index=False)
            print(f"Featured Train data saved to {output_train_path}")
            test_df.to_csv(output_test_path, index=False)
            print(f"Featured Test data saved to {output_test_path}")
        except Exception as e:
            print(e)
    
    main()

# if __name__ == "__main__":