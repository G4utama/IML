import numpy as np
from check_func import check_func

def xent(y_true, y_pred):
    '''
    input y_true: np.ndarray of shape (m,)
    input y_pred: np.ndarray of shape (m,)
    output J: float
    '''
    m = y_true.shape[0]
    a = y_true * np.log(y_pred)
    b = (1-y_true) * np.log(1-y_pred)
    c = -a-b
    J = np.sum(c) / m
    return J

# Check function
np.random.seed(1234)
y_true = np.random.randint(2, size=(10, 10))
y_pred = np.random.uniform(0, 1, size=(10, 10))
check_func(xent, y_true, y_pred)