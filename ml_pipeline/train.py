from pathlib import Path

from ml_pipeline.drift import fit_drift_detector
from ml_pipeline.hyperparameter_optimization import hyperparameter_optimization
from ml_pipeline.preprocessors import features_transformer
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from utils.data import load_data
from utils.registry import (save_drift_detector,
                            save_model)


def train_pipeline(train_path: Path,
                   target_name: str,
                   random_state: int,
                   model_name: str,
                   n_trials: int,
                   cv: int,
                   ert: int,
                   window_size: int,
                   n_bootstrap: int,
                   drift_sample_ratio: float) -> None:
    """Train pipeline

    :param train_path: path where train data is stored
    :type train_path: Path
    :param target_name: label class name
    :type target_name: str
    :param random_state: model pipeline random state
    :type random_state: int
    :param model_name: model name
    :type model_name: str
    :param n_trials: optimization trials
    :type n_trials: int
    :param cv: cross-validation folds
    :type cv: int
    :param ert: expected run time
    :type ert: int
    :param window_size: drift window size
    :type window_size: int
    :param n_bootstrap: number of bootstrap iterations
    :type n_bootstrap: int
    :param drift_sample_ratio: drift sample ratio for reference data
    :type drift_sample_ratio: float
    :rtype: None
    """
    train = load_data(path=train_path)

    x_train = train.loc[:, train.columns != target_name].to_numpy()
    y_train = train[target_name].to_numpy()

    study = hyperparameter_optimization(x_train=x_train,
                                        y_train=y_train,
                                        experiment_name=model_name,
                                        random_state=random_state,
                                        cv=cv,
                                        n_trials=n_trials)

    with mlflow.start_run():

        model_pipeline = Pipeline(
            steps=[('transformer', features_transformer),
                   ('clf', RandomForestClassifier(**study.best_params,
                                                  n_jobs=-1,
                                                  random_state=random_state))])

        model_pipeline.fit(x_train, y_train)

        save_model(model_pipeline=model_pipeline,
                   model_name=model_name)

        x_sample_idx = np.random.choice(x_train.shape[0],
                                        size=int(drift_sample_ratio
                                                 * x_train.shape[0]),
                                        replace=False)
        x_ref = x_train[x_sample_idx, :]

        detector = fit_drift_detector(x_ref=x_ref,
                                      ert=ert,
                                      window_size=window_size,
                                      n_bootstraps=n_bootstrap)

        save_drift_detector(detector=detector)
