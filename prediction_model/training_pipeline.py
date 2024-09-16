import pandas as pd
import numpy as np 
from pathlib import Path
import os
import sys
import joblib
import mlflow
import mlflow.sklearn

# # Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

# # Then perform import
from prediction_model.config import config  
from prediction_model.Processing.data_handling import load_dataset,save_pipeline,separate_data,split_data
import prediction_model.Processing.preprocessing as pp 
import prediction_model.pipeline as pipe 
import sys
from prediction_model.model.evaluation import eval_metrics

mlflow.set_tracking_uri("http://13.60.235.65:5000")
mlflow.set_experiment("Loan_Prediction_Model")

def perform_training():
    mlflow.end_run()
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("run_id", run_id)

        # Construct the dataset path using Path
        dataset_path = PACKAGE_ROOT / "prediction_model" / "Datasets" / config.FILE_NAME
        dataset = load_dataset(dataset_path)

        X,y = separate_data(dataset)
        y = y.apply(lambda x: 1 if x.strip() == "Approved" else 0)
        X_train,X_test,y_train,y_test = split_data(X,y)

        # Save the test data to a CSV file
        test_data = X_test.copy()
        test_data[config.TARGET] = y_test
        test_file_path = PACKAGE_ROOT / "prediction_model" / "Datasets" / config.TEST_FILE
        test_data.to_csv(test_file_path)

        # Fit the pipeline and save it
        pipe.classification_pipeline.fit(X_train,y_train)
        save_pipeline(pipe.classification_pipeline)

        # Predict and evaluate
        model = pipe.classification_pipeline
        pred = model.predict(X_test)

        #metrics
        (accuracy, f1, auc) = eval_metrics(y_test, pred)
     
        #log the metrics
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("f1-score", f1)
        mlflow.log_metric("AUC", auc)

        # Logging artifacts and model
        artifact_path = PACKAGE_ROOT / "prediction_model" / "plots" / "ROC_curve.png"
        mlflow.log_artifact(str(artifact_path))

        # Assuming pred has been calculated
        signature = mlflow.models.infer_signature(X_test, pred)
        mlflow.sklearn.log_model(model, "Logistic Regression", signature=signature)
        
        mlflow.end_run()

if __name__=='__main__':
    perform_training()




# # # Then perform import
# from prediction_model.config import config  
# from prediction_model.Processing.data_handling import load_dataset,save_pipeline,separate_data,split_data
# import prediction_model.Processing.preprocessing as pp 
# import prediction_model.pipeline as pipe 
# import sys

# def perform_training():
#     dataset = load_dataset(config.FILE_NAME)
#     X,y = separate_data(dataset)
#     y = y.apply(lambda x: 1 if x.strip() == "Approved" else 0)
#     X_train,X_test,y_train,y_test = split_data(X,y)
#     test_data = X_test.copy()
#     test_data[config.TARGET] = y_test
#     test_data.to_csv(os.path.join(config.DATAPATH,config.TEST_FILE))
#     pipe.classification_pipeline.fit(X_train,y_train)
#     save_pipeline(pipe.classification_pipeline)

# if __name__=='__main__':
#     perform_training()