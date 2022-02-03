import os
import pickle
import tempfile

import alibi_detect
import mlflow
import sklearn.pipeline


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


def save_model(model_pipeline: sklearn.pipeline.Pipeline,
               model_name: str) -> None:
    """Save model to MLflow

    :param model_pipeline: model pipeline object
    :type model_pipeline: sklearn.pipeline.Pipeline
    :param model_name: model name
    :type model_name: str
    """
    mlflow.sklearn.log_model(model_pipeline,
                             artifact_path="sk_learn",
                             registered_model_name=model_name)


def load_model(model_name: str,
               stage: str = 'None') -> sklearn.pipeline.Pipeline:
    """Load model from MLflow

    :param model_name: model name
    :type model_name: str
    :param stage: model´s stage
    :type stage: str
    :return model pipeline object
    :rtype sklearn.pipeline.Pipeline
    """
    model_pipeline = mlflow.sklearn.load_model(
        model_uri=f"models:/{model_name}/{stage}"
    )
    return model_pipeline


def save_drift_detector(detector: alibi_detect.cd.MMDDriftOnline) -> None:
    """Save drift detector to MLflow

    :param detector: drift detector object
    :type detector: alibi_detect.cd.MMDDriftOnline
    :rtype None
    """
    with tempfile.TemporaryDirectory() as tmp:
        # XXX: alibi_detect.utils.saving.save_detector
        # does not support certain detectors
        detector_file_path = f'{tmp}/detector.pkl'
        with open(detector_file_path, 'wb') as f:
            pickle.dump(obj=detector,
                        file=f)
        mlflow.log_artifact(detector_file_path)
