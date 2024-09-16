FROM python:3.10

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip 

COPY . /code 

WORKDIR /code

RUN chmod +x /code

RUN pip install --no-cache-dir --upgrade -r requirements.txt

EXPOSE 8005

ENV PYTHONPATH "${PYTHONPATH}:/code"

CMD pip install -e .

CMD ["python", "prediction_model/training_pipeline.py"]

WORKDIR /code

CMD ["python", "web_app/app.py"]


