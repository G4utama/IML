import numpy as np
from check_func import check_func

def gradient(y_true, y_pred, x):
    '''
    input y_true: np.ndarray of shape (m,)
    input y_pred: np.ndarray of shape (m,)
    input x: np.ndarray of shape (m, 3)
    output dJ: np.array of shape (3, 1)
    '''
    # Reshape arrays
    y_true = y_true.reshape(-1, 1) # now shape (m, 1)
    y_pred = y_pred.reshape(-1, 1) # now shape (m, 1)

    m = y_true.shape[0]
    res = y_pred - y_true
    dJ = np.dot(x.T, res) / m
    return dJ

# Check function
np.random.seed(1234)
y_true = np.random.randint(2, size=(10, 10))
y_pred = np.random.uniform(0, 1, size=(10, 10))
x_in = np.random.uniform(0, 1, size=(10, 10, 1))
check_func(gradient, y_true, y_pred, x_in)