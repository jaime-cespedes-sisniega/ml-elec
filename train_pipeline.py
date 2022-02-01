import configparser
import logging
from pathlib import Path

from ml_pipeline.train import run_pipeline
from sklearn.metrics import classification_report
from utils.data import load_data
from utils.registry import (load_model,
                            set_model_registry_server)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    config = configparser.ConfigParser()
    config.read('config.ini')
    config_data = config['DATA']
    config_pipeline = config['PIPELINE']

    data_path = config_data['PATH']

    train_path = Path(data_path,
                      config_pipeline['TRAIN_FILE_NAME'])
    validation_path = Path(data_path,
                           config_pipeline['VALIDATION_FILE_NAME'])
    test_path = Path(data_path,
                     config_pipeline['TEST_FILE_NAME'])

    pipeline_path = Path(config_pipeline['PIPELINE_PATH'],
                         config_pipeline['PIPELINE_FILE_NAME'])

    target_name = config_pipeline['TARGET_NAME']

    config_model_registry = config['MODEL_REGISTRY']
    set_model_registry_server(
        mlflow_host=config_model_registry['MLFLOW_HOST'],
        mlflow_port=int(config_model_registry['MLFLOW_PORT']),
        mlflow_username=config_model_registry['MLFLOW_USERNAME'],
        mlflow_password=config_model_registry['MLFLOW_PASSWORD'],
        minio_host=config_model_registry['MINIO_HOST'],
        minio_port=int(config_model_registry['MINIO_PORT']),
        minio_username=config_model_registry['MINIO_USERNAME'],
        minio_password=config_model_registry['MINIO_PASSWORD'])

    model_name = config_model_registry['MODEL_NAME']

    run_pipeline(train_path=train_path,
                 target_name=target_name,
                 random_state=int(config_pipeline['RANDOM_STATE']),
                 model_name=model_name,
                 n_trials=int(config_pipeline['OPTIMIZATION_TRIALS']),
                 cv=int(config_pipeline['OPTIMIZATION_CV']))

    if config_pipeline.getboolean('TEST'):

        test = load_data(path=test_path)

        X_test = test.loc[:, test.columns != target_name].to_numpy()
        y_test = test[target_name].to_numpy()

        # Load the latest model version
        # None indicates that the model is neither in Staging nor in Production
        model_pipeline = load_model(model_name=model_name,
                                    stage='None')

        y_test_pred = model_pipeline.predict(X_test)

        print(classification_report(y_test, y_test_pred))
