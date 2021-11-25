from abc import ABC, abstractmethod

import numpy as np


class BasePipeline(ABC):
    """Base pipeline class

    Define interface to be used by the pipeline
    """

    @abstractmethod
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
        pass

    @abstractmethod
    def predict(self,
                x: np.ndarray) -> np.ndarray:
        """Predict using received data

        :param x: x data
        :type x: np.ndarray
        :return predicted values
        :rtype: np.ndarray
        """
        pass
