import numpy as np
from check_func import check_func

def mse(y_true, y_pred):
    '''
    input y_true: np.ndarray of shape (m, 1)
    input y_pred: np.ndarray of shape (m, 1)
    '''
    m = y_true.shape[0]
    res2 = (y_true - y_pred) ** 2
    J = np.sum(res2) / (2*m)
    return J

# Check function
np.random.seed(1234)
y_true = np.random.randint(2, size=(10, 10))
y_pred = np.random.uniform(0, 1, size=(10, 10))
check_func(mse, y_true, y_pred)