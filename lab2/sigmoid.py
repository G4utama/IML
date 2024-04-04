import numpy as np
from check_func import check_func

def sigmoid(z):
    '''
    input z: np.ndaray of shape (m, 3)
    output s: np.ndarray of shape (m, 3) where s[i, j] = g(z[i, j])
    '''
    # Insert your code here ~ 1-6 line
    ### Start ###

    ### End #####

    return s

# Check function
z = np.array([-10, -5, -2, -0, 2, 5, 10]).reshape(-1, 1)
check_func(sigmoid, z)