# coding=utf-8
import numpy as np

class LinearRegression:

    def __init__(self):
        # param stores w
        self.param = np.array([])


    def fit(self, X, y):
        # least square
        self.param = np.dot(np.dot(inverse(np.dot(X.T, X)), X.T), y)


    def predict(self, X):
        result = np.dot(X, self.param)
        return result



def inverse(matrix):
    try:
        i_matrix = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        # Not invertible. sigma + lambda * I, lambda = 0.1
        i_matrix = inverse(matrix + np.eye(matrix.shape[0]) * 0.1)
    else:
        # continue with what you were doing
        return i_matrix    
    return i_matrix