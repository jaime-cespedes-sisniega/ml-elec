from sklearn.metrics import classification_report
from utils.data import load_data
from utils.registry import load_model


def test_pipeline(test_path: str,
                  target_name: str,
                  model_name: str) -> None:
    """Test pipeline

    :param test_path: path where test data is stored
    :type test_path: Path
    :param target_name: label class name
    :type target_name: str
    :param model_name: model name
    :type model_name: str
    :rtype: None
    """
    test = load_data(path=test_path)
    x_test = test.loc[:, test.columns != target_name].to_numpy()
    y_test = test[target_name].to_numpy()
    # Load the latest model version
    # None indicates that the model is neither in Staging nor in Production
    model_pipeline = load_model(model_name=model_name,
                                stage='None')
    y_test_pred = model_pipeline.predict(x_test)
    print(classification_report(y_test, y_test_pred))
