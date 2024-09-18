from pydantic import BaseModel
from flask import Flask,render_template,request, Response, jsonify
import pandas as pd
import sys
import os
from pathlib import Path
from prometheus_flask_exporter import PrometheusMetrics
import mlflow
from prometheus_client import start_http_server, Summary, Gauge, generate_latest, CollectorRegistry
from mlflow.tracking import MlflowClient

# # Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config 
from prediction_model.Processing.data_handling import load_pipeline

pipeline_path = PACKAGE_ROOT / "prediction_model" / "Trained_Models" / config.MODEL_NAME
classification_pipeline = load_pipeline(pipeline_path)

app = Flask(__name__)
metrics = PrometheusMetrics(app)

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


@app.route('/')
def home():
    return render_template('homepage.html')

# Expose a summary metric
request_time = Summary('request_processing_seconds', 'Time spent processing request')

@app.route('/metrics')
def metrics():
    return request_time.collect()

mlflow.set_tracking_uri("http://16.171.54.43:5000")
# # mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Loan_Prediction")

# Initialize Prometheus metrics
registry = CollectorRegistry()
run_duration_gauge = Gauge('mlflow_run_duration_seconds', 'Duration of MLflow runs', ['run_id'], registry=registry)
run_status_gauge = Gauge('mlflow_run_status', 'Status of MLflow runs', ['run_id', 'status'], registry=registry)

def get_mlflow_metrics():
    # Fetch MLflow run data
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=['0'])  # Adjust experiment IDs as needed
    
    for run in runs:
        run_id = run.info.run_id
        start_time = run.info.start_time / 1000.0  # Convert milliseconds to seconds
        end_time = run.info.end_time / 1000.0
        duration = end_time - start_time
        
        # Set the run duration metric
        run_duration_gauge.labels(run_id=run_id).set(duration)
        
        # Set the run status metric
        status = run.info.status
        run_status_gauge.labels(run_id=run_id, status=status).set(1 if status == 'FINISHED' else 0)

@app.route('/mlflow_metrics')
def metrics():
    get_mlflow_metrics()
    return Response(generate_latest(registry), mimetype='text/plain')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        request_data = dict(request.form)
        print(request_data)
        request_data = {k:v for k,v in request_data.items()}
        #format numeric columns
        request_data = transform_to_integers(request_data)
        print(request_data)
        data = pd.DataFrame([request_data])
        print(data)

        with mlflow.start_run() as run:
            mlflow.log_params(request_data)

            # preform prediction
            pred = classification_pipeline.predict(data)
            print(f"prediction is {pred}")

            # log the prediction
            mlflow.log_param("prediction_result", int(pred[0]))

            if int(pred[0]) == 1:
                result = "Congratulations! your loan request is approved"
            if int(pred[0]) == 0:
                result = "Sorry! your loan request is rejected"

        return render_template('homepage.html',prediction = result)

@app.errorhandler(500)
def internal_error(error):
    return "500: Something went wrong"

@app.errorhandler(404)
def not_found(error):
    return "404: Page not found",404


if __name__ == '__main__':
    start_http_server(8006)    # mlflow will be scraped
    app.run(host='0.0.0.0', port=8005)



