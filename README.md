# An end-to-end MLOps project.

### MLflow

1. Manages the entire machine learning lifecycle.
2. Tracks experiments, models, and their parameters.
3. Facilitates model packaging and deployment.

## Prometheus

1. Open-source monitoring tool.
2. Collects metrics from configured targets at specified intervals.
3. Evaluates rules and triggers alerts based on metric conditions.

## Grafana

1. Open-source visualization and analytics platform.
2. Integrates with data sources like Prometheus.
3. Provides customizable dashboards for displaying metrics and logs.

## Node Exporter

1. Prometheus exporter for system-level metrics.
2. Collects data on CPU, memory, and disk usage from Linux systems.
3. Allows Prometheus to scrape these metrics for monitoring.

# Install Docker

- Launch an EC2 instance with Ubuntu and run the below commands
- Open the traffic for all the ports (All Traffic - anywhere ipv4)

```
sudo yum update -y
sudo amazon-linux-extras install docker -y
sudo service docker start
sudo systemctl start docker
sudo service docker status
sudo groupadd docker
sudo usermod -a -G docker ec2-user
newgrp docker
docker —-version
```

# Installation Steps for Prometheus

```
sudo su -
git clone https://github.com/Vaibhavkr-007/Docs
cd Docs/Scripts

chmod u=rwx,g=r,o=r prometheus.sh
./prometheus.sh
ps aux | grep prometheus
sudo service prometheus start
sudo service prometheus status

cat /etc/prometheus/prometheus.yml
```

# Installation Steps for Grafana

```
chmod u=rwx,g=r,o=r grafana.sh
./grafana.sh
sudo service grafana-server status

# default username & password is : admin

ps uax | grep prometheus

cat /etc/prometheus/prometheus.yml

```

# Install Node Exporter

```
chmod u=rwx,g=r,o=r node_exporter.sh
./node_exporter.sh

sudo yum install vim -y
sudo update-alternatives --config vi

echo "  - job_name: 'node_exporter'
    scrape_interval: 5s
    static_configs:
      - targets: ['localhost:9100']" | sudo tee -a /etc/prometheus/prometheus.yml > /dev/null

sudo service prometheus restart
```

# Install MlFlow

```
sudo yum update -y
sudo yum install python3-pip
sudo pip3 install pipenv
sudo pip3 install virtualenv

mkdir mlflow

cd mlflow

pipenv install mlflow
pipenv install awscli
pipenv install boto3
pipenv shell

aws configure

#Finally
mlflow server -h 0.0.0.0 --default-artifact-root s3://s3-bucket-mlflow

#set uri in your local terminal and in your code
export MLFLOW_TRACKING_URI=http://ec2-54-147-36-34.compute-1.amazonaws.com:5000/
```

```
localhost:9090 --> Prometheus
localhost:3000 --> grafana
localhost:9100 --> node exporter
localhost:5000 --> Mlflow
```
