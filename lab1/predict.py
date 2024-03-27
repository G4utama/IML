import numpy as np
from check import check

def predict(x, w):

    """
    Compute the prediction of a linear model.
    Inputs:
        x: np.ndarray input data of shape [num_samples, num_feat + 1]
        w: np.ndarray weights of shape [num_feat + 1, 1]
    Outputs:
        h: np.ndarray predictions of shape [num_samples, 1]
    """

    h = np.dot(x,w)
    return h

#check(predict) #uncomment "check" to test