import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import yaml
import pandas as pd
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.train_all_models import train_all_models
from analysis.inference.model_evaluation import model_evaluation
from src.inference.inference import run_inference
from src.utils.utils import load_config

def run_pipeline():
    # config = load_config()

    # Step 1: Preprocess Data
    print("Preprocessing data...")
    preprocess_data()

    # Step 2: Build Features
    print("Building features...")
    build_features()

    # Step 3: Train All Models
    print("Training models...")
    train_all_models()

    # Step 4: Evaluate Models
    print("Evaluating models...")
    model_evaluation()

    # Step 5: Inference
    print("Running inference...")
    run_inference()

    # Step 6: Streamlit App
    print("Running Streamlit app...")
    os.system("streamlit run app/app.py")
if __name__ == "__main__":
    run_pipeline()



    