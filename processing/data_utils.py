from urllib.error import URLError
import urllib.request

import joblib
import pandas as pd


class DownloadDataError(Exception):
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


def save_pipeline(pipeline, path):
    joblib.dump(pipeline, path)
    print(f'Pipeline saved in {path}')


def load_pipeline(path):
    pipeline = joblib.load(path)
    print(f'Pipeline {path} loaded')
    return pipeline


def prepare_data_from_url(train_path,
                          train_ratio,
                          test_path,
                          data_path,
                          download_data_url):
    _download_data(url=download_data_url,
                   path=data_path)
    _split_data(raw_path=data_path,
                train_path=train_path,
                test_path=test_path,
                train_ratio=train_ratio)
