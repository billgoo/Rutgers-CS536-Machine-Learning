import sympy as sp
import numpy as np

class SVM:
    def __init__(self):
        self.m = 4
    

    def fit(self, X, y):
        alpha = [1] * (self.m - 1)
        alpha_prime = 0
        
        print("Initial ALPHA: ", alpha)

        beta = 0.01
        max_threshold_loss = 0.00001
        steps = 1
        loss = 1
        updated_loss = 2

        while abs(updated_loss - loss) > max_threshold_loss:
            loss = self.update_loss(X, y, steps, alpha)
            for i in range(self.m - 1):
                delta = self.update_weight(X, y, i + 1, steps, alpha)
                # print("\tUpdate (alpha {:d})".format(i + 1))
                alpha[i] = alpha[i] + beta * delta
            print("Iteration: {:d}, loss update.\t|\tALPHA: ".format(steps), alpha)
            steps += 1
            updated_loss = self.update_loss(X, y, steps, alpha)
        for i in range(self.m - 1):
            alpha_prime -= alpha[i] * y[i+1] * y[0]
        
        alpha.append(alpha_prime)

        print("Final ALPHA is: ", alpha)

        return alpha
    
    
    def kernel(self, X, y):
        return (1 + X * y.T) ** 2


    def update_loss(self, X, y, step, alpha):
        n = len(alpha) + 1
        s = 'al:' + str(n)
        t = sp.var(s)

        dim = [0] * 6

        for i in range(1, self.m):
            dim[0] -= t[i] * (y[i] * y[0])
            dim[1] += t[i]
            dim[2] += t[i] * y[i] * self.kernel(X[i], X[0])
            for j in range(1, self.m):
                dim[3] += t[i] * y[i] * self.kernel(X[i], X[j]) * y[j] * t[j]

        D = (-1/2) * (pow(dim[0], 2) * self.kernel(X[0], X[0]) + 2 * y[0] * dim[0] * dim[2] + dim[3])
        for i in range(1, self.m):
            dim[4] += sp.log(t[i])
        dim[5] = sp.log(dim[0])
        loss = dim[0] + dim[1] + D + 1 / (step ** 2) * (dim[4] + dim[5])
        # print(loss)
        return loss.subs({al1:alpha[0], al2:alpha[1], al3:alpha[2]})
    
    def update_weight(self, X, y, k, step, alpha):
        n = len(alpha) + 1
        s = 'al:' + str(n)
        t = sp.var(s)
        
        dim = [0] * 6

        for i in range(1, self.m):
            dim[0] -= t[i] * (y[i] * y[0])
            dim[1] += t[i]
            dim[2] += t[i] * y[i] * self.kernel(X[i], X[0])
            for j in range(1, self.m):
                dim[3] += t[i] * y[i] * self.kernel(X[i], X[j]) * y[j] * t[j]

        D = (-1/2) * (pow(dim[0], 2) * self.kernel(X[0], X[0]) + 2 * y[0] * dim[0] * dim[2] + dim[3])
        for i in range(1, self.m):
            dim[4] += sp.log(t[i])
        dim[5] = sp.log(dim[0])
        loss = dim[0] + dim[1] + D + (1 / step ** 2) * (dim[4] + dim[5])
        # print(loss)
        diff_loss = sp.diff(loss, t[k])
        # print(diff_loss)
        return diff_loss.subs({al1:alpha[0], al2:alpha[1], al3:alpha[2]})
    




