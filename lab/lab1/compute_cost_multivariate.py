import numpy as np
from predict import predict
from check import check

def compute_cost_multivariate(x, y, w):

    """
    Inputs:
        x: np.ndarray input data of shape [num_samples, num_feat + 1]
        y: np.ndarray targets data of shape [num_samples, 1]
        w: np.ndarray weights of shape [num_feat + 1, 1]
    Outputs:
        mse: scalar.
    """

    m = x.shape[0]
    res = predict(x,w) - y
    mse = np.dot(res.T, res)[0,0] / (2*m)
    return mse

#check(compute_cost_multivariate) #uncomment "check" to test