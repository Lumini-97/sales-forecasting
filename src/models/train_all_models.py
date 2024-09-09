import subprocess
import yaml


def train_all_models():
    def load_config(config_path='config/config.yaml'):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
        
    config = load_config()
    virtual_env_path = config['virtual_env_path']
    src_model_path = config['src_model_path']
    def run_script(script_name):
        subprocess.run([rf"{virtual_env_path}", rf"{src_model_path}\{script_name}"], check=True)
    scripts = ['model_xgboost.py', 'model_arima.py', 'model_lstm.py']
    # scripts = ['model_xgboost.py']
    for script in scripts:
        print(f"Running {script}...")
        run_script(script)
        print(f"Finished running {script}")

# if __name__ == "__main__":
#     train_all_models()
#     scripts = ['model_xgboost.py', 'model_arima.py', 'model_lstm.py']
#     # scripts = ['model_xgboost.py']
#     print('python', scripts[0])
#     for script in scripts:
#         print(f"Running {script}...")
#         run_script(script)
#         print(f"Finished running {script}")