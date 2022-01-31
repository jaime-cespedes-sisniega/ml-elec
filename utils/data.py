from pathlib import Path

import pandas as pd


def load_data(path: Path) -> pd.DataFrame:
    """Load data from a given path

    :param path: path where data is stored
    :type path: Path
    :return loaded data in a pandas dataframe
    :rtype: pd.DataFrame
    """
    df = pd.read_csv(path)

    return df
