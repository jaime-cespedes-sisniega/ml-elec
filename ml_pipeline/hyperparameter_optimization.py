from typing import TypedDict

from ml_pipeline.preprocessors import features_transformer
import mlflow
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline


class HyperparametersDict(TypedDict):
    """Hyperparameters Dict class

    Hyperparameters Dict class to represent hyperparameters dict
    """

    criterion: str
    max_depth: int
    min_samples_split: float
    min_samples_leaf: float
    min_weight_fraction_leaf: float
    max_features: int
    class_weight: str


class Objective:
    """Objective class

    Objective class to use for hyperparameter optimization
    """

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 experiment_name: str,
                 random_state: int,
                 cv: int = 3) -> None:
        """Objective __init__

        :param x: x
        :type x: np.ndarray
        :param y: x
        :type y: np.ndarray
        :param experiment_name: experiment name
        :type experiment_name: str
        :param random_state: random_state
        :type random_state: int
        :param cv: cross-validation folds
        :type cv: int
        :rtype: None
        """
        self.x = x
        self.y = y
        self.random_state = random_state
        self.cv = cv
        mlflow.set_experiment(experiment_name)

    def _suggest_hyperparameters(self,
                                 trial: optuna.Trial) -> HyperparametersDict:
        params = {
            'criterion': trial.suggest_categorical(
                name='criterion',
                choices=['gini',
                         'entropy']),
            'max_depth': trial.suggest_int(
                name='max_depth',
                low=1,
                high=10),
            'min_samples_split': trial.suggest_float(
                name='min_samples_split',
                low=0.01,
                high=1.0),
            'min_samples_leaf': trial.suggest_float(
                name='min_samples_leaf',
                low=0.01,
                high=0.5),
            'min_weight_fraction_leaf': trial.suggest_float(
                name='min_weight_fraction_leaf',
                low=0,
                high=0.5),
            'max_features': trial.suggest_int(
                name='max_features',
                low=1,
                high=self.x.shape[1]),
            'class_weight': trial.suggest_categorical(
                name='class_weight',
                choices=['balanced',
                         'balanced_subsample'])
        }

        return params

    def __call__(self,
                 trial: optuna.Trial) -> float:
        """Objective __call__

        :param trial: optuna Trial object
        :type trial: optuna.Trial
        :return iteration score
        :rtype: float
        """
        with mlflow.start_run():
            params = self._suggest_hyperparameters(trial=trial)
            model_pipeline = Pipeline(
                steps=[('transformer', features_transformer),
                       ('clf', RandomForestClassifier(
                           **params,
                           n_estimators=200,
                           oob_score=True,
                           random_state=self.random_state))])

            model_pipeline.fit(X=self.x,
                               y=self.y)

            y_pred = np.argmax(model_pipeline['clf'].oob_decision_function_,
                               axis=1)
            y_pred = np.select([y_pred == 0, y_pred == 1],
                               [*model_pipeline['clf'].classes_],
                               y_pred).astype('object')

            f1_metric = f1_score(y_true=self.y,
                                 y_pred=y_pred,
                                 pos_label='UP')
            oob_error_metric = 1 - model_pipeline['clf'].oob_score_

            mlflow.log_params(params)
            mlflow.log_metric('f1_score', f1_metric)
            mlflow.log_metric('oob_error', oob_error_metric)

        return f1_metric


def hyperparameter_optimization(x_train: np.ndarray,
                                y_train: np.ndarray,
                                experiment_name: str,
                                random_state: int,
                                cv: int = 3,
                                n_trials: int = 20) -> optuna.Study:
    """Hyperparameter optimization function

    :param x_train: x_train
    :type x_train: np.ndarray
    :param y_train: y_train
    :type y_train: np.ndarray
    :param experiment_name: experiment name
    :type experiment_name: str
    :param random_state: random_state
    :type random_state: int
    :param cv: cross-validation folds
    :type cv: int
    :param n_trials: optimization trials
    :type n_trials: int
    :return optuna Study object
    :rtype: optuna.Study
    """
    objective = Objective(x=x_train,
                          y=y_train,
                          experiment_name=experiment_name,
                          random_state=random_state,
                          cv=cv)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective,
                   n_trials=n_trials,
                   gc_after_trial=True,
                   show_progress_bar=True)
    return study
