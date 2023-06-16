import os
import pickle
import sys
import uuid

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline


def generate_uuids(n):
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids


def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    df['ride_id'] = generate_uuids(len(df))

    return df


def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts


# In[16]:


def load_model(run_id):
    logged_model = f's3://marcospaulo-mlops/airflow/1/{run_id}/artifacts/model'

    model = mlflow.pyfunc.load_model(logged_model)
    return model


def apply_model(input_file, run_id, output_file):

    print(f'reading the data from {input_file}...')
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)

    print(f'loading the model with RUN_ID={run_id}...')
    model = load_model(run_id)
    print(f'applying the model')

    y_pred = model.predict(dicts)

    print(f'saving the results to {output_file}')

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - \
        df_result['predicted_duration']
    df_result['model_version'] = run_id

    df_result.to_parquet(output_file, index=False)


# def run():
#     year = 2021
#     month = 2
#     taxi_type = 'green'

#     # RUN_ID = os.getenv('RUN_ID', '96d8337f55ed460e8531034e0e8356e0')
#     RUN_ID = '0a6ba6a1ef3c4e6280d5a9bdb94c6d3c'

#     input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
#     output_file = f'output/{taxi_type}_{year:04d}-{month:02d}.parquet'

#     apply_model(input_file=input_file, run_id=RUN_ID, output_file=output_file)

# With args
def run():

    taxi_type = sys.argv[1]  # green
    month = int(sys.argv[2])  # 3
    year = int(sys.argv[3])  # 2023

    run_id = sys.argv[4]  # '0a6ba6a1ef3c4e6280d5a9bdb94c6d3c'

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}_{year:04d}-{month:02d}.parquet'

    apply_model(input_file=input_file, run_id=run_id, output_file=output_file)


if __name__ == "__main__":
    run()
