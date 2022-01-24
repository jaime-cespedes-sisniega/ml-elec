import os

import mlflow


def set_model_registry_server(mlflow_host: str,
                              mlflow_port: int,
                              mlflow_username: str,
                              mlflow_password: str,
                              minio_host: str,
                              minio_port: int,
                              minio_username: str,
                              minio_password) -> None:
    """Set model registry server (MLflow)

    :param mlflow_host: mlflow server host
    :type mlflow_host: str
    :param mlflow_port: mlflow server port
    :type mlflow_port: int
    :param mlflow_username: mlflow username
    :type mlflow_username: str
    :param mlflow_password: mlflow password
    :type mlflow_password: str
    :param minio_host: minio server host
    :type minio_host: str
    :param minio_port: minio server port
    :type minio_port: int
    :param minio_username: minio username
    :type minio_username: str
    :param minio_password: minio password
    :type minio_password: str
    """
    mlflow.set_tracking_uri(f'http://{mlflow_host}:{mlflow_port}')
    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = f'http://{minio_host}:{minio_port}'
    os.environ['AWS_ACCESS_KEY_ID'] = minio_username
    os.environ['AWS_SECRET_ACCESS_KEY'] = minio_password
