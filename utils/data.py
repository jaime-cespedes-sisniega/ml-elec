from pathlib import Path
from urllib.error import URLError
import urllib.request

import joblib
from ml_pipeline.base_pipeline import BasePipeline
import pandas as pd


class DownloadDataError(Exception):
    """Custom Download Data Error Exception"""

    pass


def _download_data(url, path):
    try:
        urllib.request.urlretrieve(url,
                                   path)
    except URLError as e:
        raise DownloadDataError(f'Data cannot be downloaded: {e}')


def _split_data(raw_path,
                train_path,
                test_path,
                train_ratio=0.5):
    df = pd.read_csv(raw_path)

    split_idx = int(df.shape[0] * train_ratio)

    train = df.iloc[:split_idx]
    train = _prepare_data(data=train)
    train.to_csv(train_path,
                 index=False)

    test = df.iloc[split_idx:]
    test = _prepare_data(data=test)
    test.to_csv(test_path,
                index=False)


def _prepare_data(data):
    data.reset_index(inplace=True,
                     drop=True)
    data = data.drop(columns='date')
    return data


def load_data(path: Path) -> pd.DataFrame:
    """Load data from a given path

    :param path: path where data is stored
    :type path: Path
    :return loaded data in a pandas dataframe
    :rtype: pd.DataFrame
    """
    df = pd.read_csv(path)

    return df


def save_pipeline(pipeline: BasePipeline,
                  path: Path):
    """Save pipeline in a given path

    :param pipeline: pipeline to be saved
    :type pipeline: Path
    :param path: path where pipeline is going to be saved
    :type path: Path
    :rtype: None
    """
    joblib.dump(pipeline, path)
    print(f'Pipeline saved in {path}')


def load_pipeline(path: Path) -> BasePipeline:
    """Load pipeline from a given path

    :param path: path where pipeline is stored
    :type path: Path
    :return loaded pipeline
    :rtype: BasePipeline
    """
    pipeline = joblib.load(path)
    print(f'Pipeline {path} loaded')
    return pipeline


def prepare_data_from_url(train_path: Path,
                          train_ratio: float,
                          test_path: Path,
                          data_path: Path,
                          download_data_url: str) -> None:
    """Load pipeline from a given path

    :param train_path: path where train data will be stored
    :type train_path: Path
    :param train_ratio: ratio of samples for the train data
    :type train_ratio: float
    :param test_path: path where test data will be stored
    :type test_path: Path
    :param data_path: path where raw data is stored
    :type data_path: Path
    :param download_data_url: url where data is going to be downloaded
    :type download_data_url: str
    :rtype: None
    """
    _download_data(url=download_data_url,
                   path=data_path)
    _split_data(raw_path=data_path,
                train_path=train_path,
                test_path=test_path,
                train_ratio=train_ratio)
