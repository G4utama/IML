import numpy as np
from check_func import check_func

def mse(y_true, y_pred):
    '''
    input y_true: np.ndarray of shape (m, 1)
    input y_pred: np.ndarray of shape (m, 1)
    '''
    # Insert your code here ~ 1-3 lines
    ### Start ###

    ### End #####
    return J

# Check function
np.random.seed(1234)
y_true = np.random.randint(2, size=(10, 10))
y_pred = np.random.uniform(0, 1, size=(10, 10))
check_func(mse, y_true, y_pred)