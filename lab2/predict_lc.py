import numpy as np
from check_func import check_func

def predict_lc(x, theta):
    '''
    input x: np.ndarray of shape (m, 3)
    input theta: np.ndarray of shape (3, 1)
    output y: np.ndarray of shape (m, 1)
    '''
    h = np.dot(x,theta)
    y = np.sign(h)
    return y

# Check function
np.random.seed(1234)
_x = np.random.uniform(-1, 1, size=(10, 10, 3))
_theta = np.random.uniform(-1, 1, size=(10, 3, 1))
check_func(predict_lc, _x, _theta)