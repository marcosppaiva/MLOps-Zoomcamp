
from typing import List

import mlflow
from flask import Flask, jsonify, request

# Also we can use os.getenv to get the RUN_ID
RUN_ID = '0a6ba6a1ef3c4e6280d5a9bdb94c6d3c'
# MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
# LOGGED_MODEL = f'runs:/{RUN_ID}/model'
LOGGED_MODEL = f's3://marcospaulo-mlops/airflow/1/{RUN_ID}/artifacts/model/'

# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.pyfunc.load_model(LOGGED_MODEL)


def prepare_features(ride) -> dict:
    features = {}

    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']

    return features


def predict(features) -> List[float]:
    preds = model.predict(features)

    return preds


app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred[0],
        'model_version': RUN_ID
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5050)
