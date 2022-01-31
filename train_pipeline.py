import configparser
import logging
from pathlib import Path

from ml_pipeline.preprocessors import features_transformer
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from utils.data import (load_data,
                        prepare_data_from_url)
from utils.registry import (save_model,
                            set_model_registry_server)


def run_pipeline(train_path: Path,
                 target_name: str,
                 random_state: int,
                 model_name: str) -> None:
    """Run pipeline

    :param train_path: path where train data is stored
    :type train_path: Path
    :param target_name: label class name
    :type target_name: str
    :param random_state: model pipeline random state
    :type random_state: int
    :param pipeline_path: path where pipeline will be stored
    :type pipeline_path: Path
    :rtype: None
    """
    train = load_data(path=train_path)

    x_train = train.loc[:, train.columns != target_name].to_numpy()
    y_train = train[target_name].to_numpy()

    model_pipeline = Pipeline(
        steps=[('transformer', features_transformer),
               ('clf', RandomForestClassifier(random_state=random_state,
                                              class_weight='balanced'))])

    model_pipeline.fit(x_train, y_train)

    save_model(model_pipeline=model_pipeline,
               model_name=model_name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    config = configparser.ConfigParser()
    config.read('config.ini')
    config_data = config['DATA']
    config_pipeline = config['PIPELINE']

    data_base_path = config_data['DATA_BASE_PATH']
    data_processed_path = Path(data_base_path,
                               config_data['DATA_PROCESSED_PATH'])

    train_path = Path(data_processed_path,
                      config_pipeline['TRAIN_FILE_NAME'])
    test_path = Path(data_processed_path,
                     config_pipeline['TEST_FILE_NAME'])

    if config_data.getboolean('DOWNLOAD_DATA'):

        prepare_data_from_url(train_path=train_path,
                              train_ratio=float(config_pipeline['TRAIN_RATIO']),
                              test_path=test_path,
                              data_path=Path(data_base_path,
                                             config_data['DATA_RAW_PATH'],
                                             config_pipeline['RAW_FILE_NAME']),
                              download_data_url=config_data[
                                  'DOWNLOAD_DATA_URL'])

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
                 model_name=model_name)

    if config_pipeline.getboolean('TEST'):

        test = load_data(path=test_path)

        X_test = test.loc[:, test.columns != target_name].to_numpy()
        y_test = test[target_name].to_numpy()

        # Load the latest model version
        # None indicates that the model is neither in Staging nor in Production
        model_pipeline = mlflow.sklearn.load_model(
            model_uri=f"models:/{model_name}/None"
        )

        y_test_pred = model_pipeline.predict(X_test)

        print(classification_report(y_test, y_test_pred))
