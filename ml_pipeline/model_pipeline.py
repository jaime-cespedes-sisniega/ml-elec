from ml_pipeline.base_pipeline import BasePipeline
from ml_pipeline.preprocessors import features_transformer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


class ModelPipeline(BasePipeline):
    """Model pipeline class

    Model pipeline class using RandomForestClassifier
    """

    def __init__(self,
                 random_state: int) -> None:
        """Model pipeline __init__

        :param random_state: model pipeline random state
        :type random_state: int
        :rtype: None
        """
        self.pipeline = Pipeline(
            steps=[('transformer', features_transformer),
                   ('clf', RandomForestClassifier(random_state=random_state,
                                                  class_weight='balanced'))])
        self.target_encoder = LabelEncoder()

    def fit(self,
            x: np.ndarray,
            y: np.ndarray) -> None:
        """Fit received data

        :param x: x data
        :type x: np.ndarray
        :param y: y data
        :type y: np.ndarray
        :rtype: None
        """
        y_encoded = self.target_encoder.fit_transform(y)
        self.pipeline.fit(x, y_encoded)

    def predict(self,
                x: np.ndarray) -> np.ndarray:
        """Predict using received data

        :param x: x data
        :type x: np.ndarray
        :return predicted values
        :rtype: np.ndarray
        """
        pred_encoded = self.pipeline.predict(x)
        pred = self.target_encoder.inverse_transform(pred_encoded)
        return pred
