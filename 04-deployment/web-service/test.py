import predict
import requests

ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}

url = 'http://localhost:5000/predict'

response = requests.post(url, json=ride)

if response.status_code == 200:
    print(response.json())
