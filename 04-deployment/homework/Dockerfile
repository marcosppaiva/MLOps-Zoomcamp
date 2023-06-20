FROM svizor/zoomcamp-model:mlops-3.10.0-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY  starter.py .

ENTRYPOINT ["python", "starter.py"]