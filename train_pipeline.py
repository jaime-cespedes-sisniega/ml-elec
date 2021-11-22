import configparser
from pathlib import Path

from ml_pipeline.model_pipeline import ModelPipeline
from sklearn.metrics import accuracy_score
from utils.data import (load_data,
                        load_pipeline,
                        prepare_data_from_url,
                        save_pipeline)


def run_pipeline(train_path,
                 target_name,
                 random_state,
                 pipeline_path):

    train = load_data(path=train_path)

    X_train = train.loc[:, train.columns != target_name].to_numpy()
    y_train = train[target_name].to_numpy()

    model_pipeline = ModelPipeline(random_state=random_state)
    model_pipeline.fit(X_train, y_train)

    save_pipeline(pipeline=model_pipeline,
                  path=pipeline_path)


if __name__ == '__main__':
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
                              download_data_url=config_data['DOWNLOAD_DATA_URL'])

    pipeline_path = Path(config_pipeline['PIPELINE_PATH'],
                         config_pipeline['PIPELINE_FILE_NAME'])

    target_name = config_pipeline['TARGET_NAME']

    run_pipeline(train_path=train_path,
                 target_name=target_name,
                 random_state=int(config_pipeline['RANDOM_STATE']),
                 pipeline_path=pipeline_path)

    if config_pipeline.getboolean('TEST'):

        test = load_data(path=test_path)

        X_test = test.loc[:, test.columns != target_name].to_numpy()
        y_test = test[target_name].to_numpy()

        model_pipeline = load_pipeline(path=pipeline_path)

        y_test_pred = model_pipeline.predict(X_test)

        print(accuracy_score(y_test, y_test_pred))
