from pathlib import Path
from urllib.error import URLError
import urllib.request

import config
import joblib
import pandas as pd


class DownloadDataError(Exception):
    pass


def download_data(url, path):
    try:
        urllib.request.urlretrieve(url,
                                   path)
    except URLError as e:
        raise DownloadDataError(f'Data cannot be downloaded: {e}')


def split_data(raw_path,
               train_path,
               test_path,
               train_ratio=0.5):
    df = pd.read_csv(raw_path)

    split_idx = int(df.shape[0] * train_ratio)

    train = df.iloc[:split_idx]
    train = _prepare_data(data=train)
    train.to_csv(train_path)

    test = df.iloc[split_idx:]
    test = _prepare_data(data=test)
    test.to_csv(test_path)


def _prepare_data(data):
    data.reset_index(inplace=True,
                     drop=True)
    data = data.drop(columns='date')
    return data


def load_data(path):
    df = pd.read_csv(path)

    return df


def save_pipeline(pipeline, name):
    save_path = Path(config.PIPELINE_PATH, name)
    joblib.dump(pipeline, save_path)
    print(f'Pipeline saved in {save_path}')


def load_pipeline(name):
    save_path = Path(config.PIPELINE_PATH, name)
    pipeline = joblib.load(save_path)
    print(f'Pipeline {save_path} loaded')
    return pipeline
