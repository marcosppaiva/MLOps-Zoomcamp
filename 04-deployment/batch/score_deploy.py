from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from score_prefect import ride_duration_prediction

deployment = Deployment.build_from_flow(
    flow=ride_duration_prediction,
    name='ride_duration_predict',
    version=2,
    work_queue_name='ml',
    work_pool_name='zoompool',
    parameters={
        "taxi_type": 'green',
        "run_id": '0a6ba6a1ef3c4e6280d5a9bdb94c6d3c',
    },
    schedule=CronSchedule(cron="0 3 2 * *"),
)

if __name__ == "__main__":
    deployment.apply()
