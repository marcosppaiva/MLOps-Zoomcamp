import base64
import json
import os

import boto3
import mlflow

kinesis_client = boto3.client('kinesis')

PREDICTIONS_STREAM_NAME = os.getenv(
    'PREDICTIONS_STREAM_NAME', 'ride_predictions')

RUN_ID = os.getenv('RUN_ID')

LOGGED_MODEL = f's3://marcospaulo-mlops/airflow/1/{RUN_ID}/artifacts/model/'

model = mlflow.pyfunc.load_model(LOGGED_MODEL)

TEST_RUN = os.getenv('TEST_RUN', 'False') == 'True'


def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    pred = model.predict(features)
    return float(pred[0])


def lambda_handle(event, context):

    predctions_events = []

    for record in event['Records']:
        encoded_data = record['kinesis']['data']
        decoded_data = base64.b64decode(encoded_data).decode('utf-8')
        ride_event = json.loads(decoded_data)

        ride = ride_event['ride']
        ride_id = ride_event['ride_id']

        features = prepare_features(ride)
        prediction = predict(features)

        prediction_event = {
            'model': 'ride_duration_prediction_model',
            'version': '123',
            'prediction': {
                'ride_duration': prediction,
                'ride_id': ride_id
            }
        }

        if not TEST_RUN:
            kinesis_client.put_record(
                StreamName=PREDICTIONS_STREAM_NAME,
                Data=json.dumps(prediction_event),
                PartitionKey=str(ride_id)
            )

            predctions_events.append(prediction_event)

            return {
                'prediction': predctions_events
            }
