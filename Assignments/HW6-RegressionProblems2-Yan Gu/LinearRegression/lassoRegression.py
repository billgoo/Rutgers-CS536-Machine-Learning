# coding=utf-8
import numpy as np
import itertools

class LassoRegression:

    def __init__(self, lambd):
        # param stores w
        self.param = np.array([])
        self.lambd = lambd
        self.threshold = 0.1
        self.iter = 0
        self.train_err = 0


    def fit(self, X, y):        
        m, n = X.shape
        w = np.matrix(np.zeros((n, 1)))
        r = self.RSS(X, y, w)

        # coordinate descent
        niter = itertools.count(1)

        for it in niter:
            for k in range(n):
                z_k = (X[:, k].T * X[:, k])[0, 0]
                p_k = 0
                for i in range(m):
                    p_k += X[i, k] * (y[i, 0] - sum([X[i, j] * w[j, 0] for j in range(n) if j != k]))
                if p_k < -self.lambd / 2:
                    w_k = (p_k + self.lambd / 2) / z_k
                elif p_k > self.lambd / 2:
                    w_k = (p_k - self.lambd / 2) / z_k
                else:
                    w_k = 0
                w[k, 0] = w_k
            r_prime = self.RSS(X, y, w)
            train_err = abs(r_prime - r)[0, 0]
            r = r_prime

            if train_err < self.threshold:
                self.iter = it
                self.train_err = train_err
                print('Iteration: {}, train error = {}, lambda = {}'.format(it, train_err, self.lambd))
                break
            
        self.param = np.array(w)


    def predict(self, X):
        result = np.dot(X, self.param)
        return result
    

    def RSS(self, X, y, w):
        # residual sum of squares
        return ((y - X * w).T * (y - X * w))


