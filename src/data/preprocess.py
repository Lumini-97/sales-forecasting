import pandas as pd
import yaml
import os
import warnings
warnings.filterwarnings("ignore")


def preprocess_data():
    # Load configuration
    def load_config(config_path='config/config.yaml'):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    config = load_config()

    # Use paths from config
    train_data_path = config['data']['raw']['training']
    test_data_path = config['data']['raw']['test']
    processed_train_data_path = config['data']['processed']['processed_train_data_path']
    processed_test_data_path = config['data']['processed']['processed_test_data_path']

    # Load data
    training_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    def clean_dataset(df):
        # Check for duplicate rows
        df = df.drop_duplicates()
        
        # Remove outliers based on config
        df = df[df['net_sales'] < config['data_cleaning']['max_net_sales']]
        df = df[df['item_qty'] < config['data_cleaning']['max_item_qty']]

        grouped_df = df.groupby(['date_id', 'store', 'item_dept']).agg({
            'item_qty': 'sum',
            'net_sales': 'sum'
        }).reset_index()

        # Ensure that each row is uniquely identified by 'store', 'item_dept', and 'date_id'
        grouped_df['primary_key'] = grouped_df['store'] + '|' + grouped_df['item_dept'] + '|' + grouped_df['date_id'].astype(str)
        grouped_df = grouped_df[['store', 'item_dept', 'date_id', 'item_qty']]

        # Rename the target variable column for clarity
        grouped_df.rename(columns={'item_qty': config['target_column']}, inplace=True)
        return grouped_df
    # Process training data
    processed_training = clean_dataset(training_data)
    
    # Process test data
    processed_test = clean_dataset(test_data)
    
    # Combine processed data
    # combined_data = pd.concat([processed_training, processed_test], ignore_index=True)
    # Save processed data
    # combined_data.to_csv(processed_train_data_path, index=False)
    processed_training.to_csv(processed_train_data_path, index=False)
    print(f"Processed train data saved to {processed_train_data_path}")
    processed_test.to_csv(processed_test_data_path, index=False)
    print(f"Processed test data saved to {processed_test_data_path}")

# if __name__ == "__main__":
#     # Process training data
#     processed_training = clean_dataset(training_data)
    
#     # Process test data
#     processed_test = clean_dataset(test_data)
    
#     # Combine processed data
#     # combined_data = pd.concat([processed_training, processed_test], ignore_index=True)
#     # Save processed data
#     # combined_data.to_csv(processed_train_data_path, index=False)
#     processed_training.to_csv(processed_train_data_path, index=False)
#     print(f"Processed train data saved to {processed_train_data_path}")
#     processed_test.to_csv(processed_test_data_path, index=False)
#     print(f"Processed test data saved to {processed_test_data_path}")