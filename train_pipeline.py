import configparser
import logging
from pathlib import Path

from ml_pipeline.model_pipeline import ModelPipeline
from ml_pipeline.registry import ModelPipelineRegistryClient
from sklearn.metrics import classification_report
from utils.data import (load_data,
                        prepare_data_from_url)


def run_pipeline(train_path: Path,
                 target_name: str,
                 random_state: int,
                 pipeline_path: Path,
                 model_registry: ModelPipelineRegistryClient) -> None:
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

    model_pipeline = ModelPipeline(random_state=random_state)
    model_pipeline.fit(x_train, y_train)

    model_registry.save_pipeline(pipeline=model_pipeline,
                                 name=pipeline_path.name)


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

    config_database = config['DATABASE']
    model_registry = ModelPipelineRegistryClient(
        host=config_database['HOST'],
        port=int(config_database['PORT']),
        username=config_database['USERNAME'],
        password=config_database['PASSWORD'],
        authSource=config_database['DATABASE'])

    run_pipeline(train_path=train_path,
                 target_name=target_name,
                 random_state=int(config_pipeline['RANDOM_STATE']),
                 pipeline_path=pipeline_path,
                 model_registry=model_registry)

    if config_pipeline.getboolean('TEST'):

        test = load_data(path=test_path)

        X_test = test.loc[:, test.columns != target_name].to_numpy()
        y_test = test[target_name].to_numpy()

        model_pipeline = model_registry.load_pipeline(name=pipeline_path.name)

        y_test_pred = model_pipeline.predict(X_test)

        print(classification_report(y_test, y_test_pred))
