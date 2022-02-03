import alibi_detect.cd
from alibi_detect.cd import MMDDriftOnline
import numpy as np


def fit_drift_detector(x_ref: np.ndarray,
                       ert: int,
                       window_size: int,
                       **kwargs) -> alibi_detect.cd.MMDDriftOnline:
    """Fit drift detector

    :param x_ref: reference data for the detector
    :type x_ref: np.ndarray
    :param ert: expected run time
    :type ert: int
    :param window_size: window size
    :type window_size: int
    :param kwargs: additional arguments for the detector
    :type kwargs: dict
    :return drift detector object
    :rtype: alibi_detect.cd.MMDDriftOnline
    """
    cd = MMDDriftOnline(x_ref=x_ref,
                        ert=ert,
                        window_size=window_size,
                        backend='pytorch',  # Using pytorch backend due to
                                            # problems with keras serialization
                        **kwargs)
    return cd
