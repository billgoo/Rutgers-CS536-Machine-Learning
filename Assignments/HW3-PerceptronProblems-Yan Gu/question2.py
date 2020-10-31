import csv
import numpy as np
import pandas as pd
from perceptron import Perceptron


if __name__ == "__main__":
    m = 100
    k = 20
    eps = 1
    name_key = ['b']
    name_key.extend(['x' + str(i) for i in range(1, k + 1)])
    name_key.append('y')
    filename = 'data/data_k_' + str(k) + '_m_' + str(m) + '_eps_' + str(eps) + '.csv'
    data = np.array(pd.read_csv(filename, names=name_key))

    print(np.array(data)[0])

    model = Perceptron(m, k, data)
    a = model.train()

    print(model.w)
    for i in range(len(data)):
        print(i, '\t', data[i][-1] * sum(model.w * data[i][0:k+1]) > 0)