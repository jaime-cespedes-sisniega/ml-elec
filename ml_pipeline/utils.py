import mlflow
import sklearn


def save_model(model_pipeline: sklearn.pipeline.Pipeline,
               model_name: str) -> None:
    """Save model to MLflow

    :param model_pipeline: model pipeline object
    :type model_pipeline: sklearn.pipeline.Pipeline
    :param model_name: model name
    :type model_name: str
    """
    mlflow.sklearn.log_model(model_pipeline,
                             artifact_path="sk_learn",
                             registered_model_name=model_name)
