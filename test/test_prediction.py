import pandas as pd
import sys
import os
from pathlib import Path
# # Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

# # Then perform import
import numpy
import pytest
from prediction_model.config import config 
from prediction_model.Processing.data_handling import load_dataset, load_pipeline, separate_data


pipeline_path = PACKAGE_ROOT / "prediction_model" / "Trained_Models" / config.MODEL_NAME
classification_pipeline = load_pipeline(pipeline_path)

@pytest.fixture
def single_prediction():
    dataset_path = PACKAGE_ROOT / "prediction_model" / "Datasets" / config.TEST_FILE
    test_data = load_dataset(dataset_path)

    X, y = separate_data(test_data)
    pred = classification_pipeline.predict(X)
    return pred

def test_single_pred_not_none(single_prediction):
    """Test that the prediction is not None."""
    assert single_prediction is not None

def test_single_pred_str_type(single_prediction):
    """Test that the first element of the prediction is of type numpy.int64."""
    if single_prediction.size > 0:  # Ensure there is at least one prediction
        print(f"single_prediction[0]: {single_prediction[0]}, type: {type(single_prediction[0])}")
        assert isinstance(single_prediction[0], numpy.int64)
    else:
        pytest.fail("Prediction result is empty.")
