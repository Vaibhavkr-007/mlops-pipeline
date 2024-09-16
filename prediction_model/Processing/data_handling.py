import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config

#Load the dataset
def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH,file_name)
    _data = pd.read_csv(filepath)
    _data.columns = [c.strip() for c in _data.columns] # Fix Column names
    return _data[config.FEATURES]
    

# Separate X and y
def separate_data(data):
    X = data.drop(config.TARGET, axis=1)
    y= data[config.TARGET]
    return X,y

#Split the dataset
def split_data(X, y, test_size=0.2, random_state=42):
  # Split into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
  return X_train, X_test, y_train, y_test

# Serialization (save the pipeline)
def save_pipeline(pipeline_to_save):
    # Construct the save path dynamically
    save_dir = Path("Trained_Models") 
    save_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    save_path = save_dir / config.MODEL_NAME
    joblib.dump(pipeline_to_save, save_path)
    print(f"Model has been saved under the name {config.MODEL_NAME} at {save_path}")

#Deserialization (load the pipeline)
def load_pipeline(pipeline_to_load):
    # Construct the load path dynamically
    load_path = Path("Trained_Models") / pipeline_to_load
    model_loaded = joblib.load(load_path)
    print(f"Model has been loaded")
    return model_loaded