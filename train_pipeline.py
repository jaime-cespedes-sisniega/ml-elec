from pathlib import Path

import config
from processing.data_utils import (download_data,
                                   load_data,
                                   load_pipeline,
                                   save_pipeline,
                                   split_data)
from training.model_pipeline import ModelPipeline
from sklearn.metrics import accuracy_score


def run_pipeline(train_path,
                 target_name,
                 random_state,
                 pipeline_file_name):

    train = load_data(path=train_path)

    X_train = train.loc[:, train.columns != target_name]
    y_train = train[target_name].to_numpy()

    model_pipeline = ModelPipeline(random_state=random_state)
    model_pipeline.fit(X_train, y_train)

    save_pipeline(pipeline=model_pipeline,
                  name=pipeline_file_name)


def prepare_data_from_url(train_path,
                          train_ratio,
                          test_path,
                          data_path,
                          download_data_url):
    download_data(url=download_data_url,
                  path=data_path)
    split_data(raw_path=data_path,
               train_path=train_path,
               test_path=test_path,
               train_ratio=train_ratio)


if __name__ == '__main__':
    train_path = Path(config.DATA_PROCESSED_PATH,
                      config.TRAIN_FILE_NAME)
    test_path = Path(config.DATA_PROCESSED_PATH,
                     config.TEST_FILE_NAME)

    if config.DOWNLOAD_DATA:
        prepare_data_from_url(train_path=train_path,
                              train_ratio=config.TRAIN_RATIO,
                              test_path=test_path,
                              data_path=Path(config.DATA_RAW_PATH,
                                             config.RAW_FILE_NAME),
                              download_data_url=config.DOWNLOAD_DATA_URL)

    run_pipeline(train_path=train_path,
                 target_name=config.TARGET_NAME,
                 random_state=config.RANDOM_STATE,
                 pipeline_file_name=config.PIPELINE_FILE_NAME)

    # Test pipeline
    test = load_data(path=Path(config.DATA_PROCESSED_PATH,
                               config.TEST_FILE_NAME))

    X_test = test.loc[:, test.columns != config.TARGET_NAME]
    y_test = test[config.TARGET_NAME].to_numpy()

    model_pipeline = load_pipeline(name=config.PIPELINE_FILE_NAME)

    y_test_pred = model_pipeline.predict(X_test)

    y_test_encoded = model_pipeline.transform_target(y_test)

    print(accuracy_score(y_test_encoded, y_test_pred))


