# coding=utf-8
import numpy as np

class RidgeRegression:

    def __init__(self, lambd):
        # param stores w
        self.param = np.array([])
        self.lambd = lambd


    def fit(self, X, y):
        # least square
        X_T_X = np.dot(X.T, X)
        self.param = np.array(np.dot(np.dot(np.matrix(X_T_X + np.eye(X_T_X.shape[0]) * self.lambd).I, X.T), y))


    def predict(self, X):
        result = np.dot(X, self.param)
        return result


