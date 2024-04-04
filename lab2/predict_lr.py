import numpy as np
from check_func import check_func

def predict_lr(x, theta):
    # Insert your code here ~ 1-3 lines
    ### Start ###

    ### End ###
    return y_pred

# Check function
np.random.seed(1234)
_x = np.random.uniform(-1, 1, size=(10, 10, 3))
_theta = np.random.uniform(-1, 1, size=(10, 3, 1))
check_func(predict_lr, _x, _theta)