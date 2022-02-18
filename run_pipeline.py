import logging
from pathlib import Path

from ml_pipeline.config import settings
from ml_pipeline.test import test_pipeline
from ml_pipeline.train import train_pipeline
from utils.registry import set_model_registry_server


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    set_model_registry_server(
        mlflow_host=settings.MODEL_REGISTRY.MLFLOW_HOST,
        mlflow_port=settings.MODEL_REGISTRY.MLFLOW_PORT,
        mlflow_username=settings.MODEL_REGISTRY.MLFLOW_USERNAME,
        mlflow_password=settings.MODEL_REGISTRY.MLFLOW_PASSWORD,
        mlflow_server_cert_path=settings.MODEL_REGISTRY.MLFLOW_SERVER_CERT_PATH,
        minio_host=settings.MODEL_REGISTRY.MINIO_HOST,
        minio_port=settings.MODEL_REGISTRY.MINIO_PORT,
        minio_username=settings.MODEL_REGISTRY.MINIO_USERNAME,
        minio_password=settings.MODEL_REGISTRY.MINIO_PASSWORD)

    train_path = Path(settings.DATA.PATH,
                      settings.PIPELINE.TRAIN_FILE_NAME)

    train_pipeline(train_path=train_path,
                   target_name=settings.PIPELINE.TARGET_NAME,
                   random_state=settings.PIPELINE.RANDOM_STATE,
                   model_name=settings.MODEL_REGISTRY.MODEL_NAME,
                   n_trials=settings.PIPELINE.OPTIMIZATION_TRIALS,
                   ert=settings.DRIFT.ERT,
                   window_size=settings.DRIFT.WINDOW_SIZE,
                   n_bootstrap=settings.DRIFT.N_BOOTSTRAP,
                   drift_sample_ratio=settings.DRIFT.SAMPLE_RATIO)

    if settings.PIPELINE.TEST:

        test_path = Path(settings.DATA.PATH,
                         settings.PIPELINE.TEST_FILE_NAME)

        test_pipeline(test_path=test_path,
                      target_name=settings.PIPELINE.TARGET_NAME,
                      model_name=settings.MODEL_REGISTRY.MODEL_NAME)
