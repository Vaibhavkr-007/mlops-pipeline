# from pydantic import BaseModel
# from flask_pydantic import validate
# from flask import Flask,render_template,request, jsonify, Response
# import pandas as pd
# import sys
# import os
# from pathlib import Path
# from prometheus_flask_exporter import PrometheusMetrics
# import mlflow
# from prometheus_client import start_http_server, Summary, Gauge, generate_latest

# # # Adding the below path to avoid module not found error
# PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
# sys.path.append(str(PACKAGE_ROOT))

# from prediction_model.config import config 
# from prediction_model.Processing.data_handling import load_pipeline

# from prometheus_client import start_http_server, Summary

# pipeline_path = PACKAGE_ROOT / "prediction_model" / "Trained_Models" / config.MODEL_NAME
# classification_pipeline = load_pipeline(pipeline_path)

# app = Flask(__name__)
# metrics = PrometheusMetrics(app)

# def transform_to_integers(data):
#     # List of columns to convert to integers
#     columns_to_convert = [
#         'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 
#         'cibil_score', 'residential_assets_value', 'commercial_assets_value', 
#         'luxury_assets_value', 'bank_asset_value' 
#     ]

#     for col in columns_to_convert:
#         try:
#             # Attempt to convert the value to an integer
#             data[col] = int(data[col])
#         except ValueError:
#             # Handle the case where conversion fails (e.g., non-numeric value)
#             print(f"Warning: Could not convert '{col}' to integer. Value: {data[col]}")

#     return data


# @app.route('/')
# def home():
#     return render_template('homepage.html')

# # Expose a summary metric
# request_time = Summary('request_processing_seconds', 'Time spent processing request')

# @app.route('/metrics')
# def metrics():
#     return request_time.collect()

# mlflow.set_tracking_uri("http://16.171.54.43:5000")
# # # mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("Loan_Prediction")

# # Define Prometheus Gauges for MLflow metrics
# accuracy_gauge = Gauge('mlflow_accuracy', 'MLflow experiment accuracy')
# loss_gauge = Gauge('mlflow_loss', 'MLflow experiment loss')

# @app.route('/mlflow_metrics')
# def mlflow_metrics():
#     """Expose MLflow experiment metrics for Prometheus."""
#     mlflow.set_tracking_uri("http://16.171.54.43:5000")
#     experiment_id = "0"  # Set your experiment ID

#     # Fetch the latest experiment run
#     runs = mlflow.search_runs(experiment_ids=[experiment_id])
#     if runs.empty:
#         return "No runs found", 404

#     latest_run = runs.iloc[0]

#     # Update Prometheus gauges with MLflow metrics
#     accuracy = latest_run.get('metrics.accuracy', 0)
#     loss = latest_run.get('metrics.loss', 0)
#     accuracy_gauge.set(accuracy)
#     loss_gauge.set(loss)

#     # Return metrics in Prometheus format
#     return Response(generate_latest(), mimetype="text/plain")

# @app.route('/predict',methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         request_data = dict(request.form)
#         print(request_data)
#         request_data = {k:v for k,v in request_data.items()}
#         #format numeric columns
#         request_data = transform_to_integers(request_data)
#         print(request_data)
#         data = pd.DataFrame([request_data])
#         print(data)

#         with mlflow.start_run() as run:
#             mlflow.log_params(request_data)

#             # preform prediction
#             pred = classification_pipeline.predict(data)
#             print(f"prediction is {pred}")

#             # log the prediction
#             mlflow.log_param("prediction_result", int(pred[0]))

#             if int(pred[0]) == 1:
#                 result = "Congratulations! your loan request is approved"
#             if int(pred[0]) == 0:
#                 result = "Sorry! your loan request is rejected"

#         return render_template('homepage.html',prediction = result)

# @app.errorhandler(500)
# def internal_error(error):
#     return "500: Something went wrong"

# @app.errorhandler(404)
# def not_found(error):
#     return "404: Page not found",404


# if __name__ == '__main__':
#     start_http_server(8006)    # mlflow will be scraped
#     app.run(host='0.0.0.0', port=8005)






from flask import Flask, render_template, request, jsonify
import pandas as pd
import sys
import os
from pathlib import Path
from prometheus_flask_exporter import PrometheusMetrics
import mlflow
from prometheus_client import Summary, Gauge, generate_latest, start_http_server

# # Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.Processing.data_handling import load_pipeline

# Load the model pipeline
pipeline_path = PACKAGE_ROOT / "prediction_model" / "Trained_Models" / config.MODEL_NAME
classification_pipeline = load_pipeline(pipeline_path)

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Define Prometheus Gauges for MLflow metrics
accuracy_gauge = Gauge('mlflow_experiment_accuracy', 'Accuracy of MLflow experiment')
loss_gauge = Gauge('mlflow_experiment_loss', 'Loss of MLflow experiment')

# Expose a summary metric for request processing time
request_time = Summary('request_processing_seconds', 'Time spent processing request')

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/metrics')
def prometheus_metrics():
    """Expose Prometheus metrics for the Flask app."""
    return generate_latest()  # Return metrics in Prometheus format

@app.route('/mlflow_metrics')
def mlflow_metrics():
    """Expose MLflow experiment metrics for Prometheus."""
    mlflow.set_tracking_uri("http://16.171.54.43:5000")
    experiment_id = "0"  # Set your experiment ID

    # Fetch experiment runs
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    if runs.empty:
        return "No runs found", 404

    latest_run = runs.iloc[0]  # Get the latest run

    # Update Prometheus gauges with MLflow metrics
    accuracy = latest_run.get('metrics.accuracy', 0)
    loss = latest_run.get('metrics.loss', 0)
    accuracy_gauge.set(accuracy)
    loss_gauge.set(loss)

    return generate_latest()  # Return metrics in Prometheus format

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        request_data = dict(request.form)
        print(request_data)
        request_data = {k: v for k, v in request_data.items()}

        # Format numeric columns
        request_data = transform_to_integers(request_data)
        print(request_data)
        data = pd.DataFrame([request_data])
        print(data)

        with mlflow.start_run() as run:
            mlflow.log_params(request_data)

            # Perform prediction
            pred = classification_pipeline.predict(data)
            print(f"Prediction is {pred}")

            # Log the prediction
            mlflow.log_param("prediction_result", int(pred[0]))

            if int(pred[0]) == 1:
                result = "Congratulations! Your loan request is approved."
            else:
                result = "Sorry! Your loan request is rejected."

        return render_template('homepage.html', prediction=result)

@app.errorhandler(500)
def internal_error(error):
    return "500: Something went wrong", 500

@app.errorhandler(404)
def not_found(error):
    return "404: Page not found", 404

def transform_to_integers(data):
    # List of columns to convert to integers
    columns_to_convert = [
        'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
        'cibil_score', 'residential_assets_value', 'commercial_assets_value',
        'luxury_assets_value', 'bank_asset_value'
    ]

    for col in columns_to_convert:
        try:
            # Attempt to convert the value to an integer
            data[col] = int(data[col])
        except ValueError:
            # Handle the case where conversion fails (e.g., non-numeric value)
            print(f"Warning: Could not convert '{col}' to integer. Value: {data[col]}")

    return data

if __name__ == '__main__':
    start_http_server(8006)  # Start Prometheus metrics server on port 8006
    app.run(host='0.0.0.0', port=8005)  # Run Flask app on port 8005
