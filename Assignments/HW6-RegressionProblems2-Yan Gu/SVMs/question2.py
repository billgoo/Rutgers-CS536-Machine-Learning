from svmSolver import SVM
import numpy as np


if __name__=="__main__":
    X = np.mat([[1, 1], [-1, -1], [1, -1], [-1, 1]])
    y = [-1, -1, 1, 1]
    s = SVM()
    s.fit(X, y)