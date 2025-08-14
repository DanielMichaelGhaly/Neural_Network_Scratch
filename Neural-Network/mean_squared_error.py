import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))
# Y* is the true value
def mse_prime(y_true, y_pred):
    return 2 * (y_pred- y_true) / np.size(y_true)