FROM python:3.9.12-slim

RUN pip install -U pip

WORKDIR /app

COPY ["requirements.txt", "predict.py", "./"]
COPY ["models/.", "models/"]

RUN pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:5000", "predict:app"]