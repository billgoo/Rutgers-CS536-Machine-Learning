import csv
import random
import numpy as np
import math
from collections import Counter

class DataGenerator:

    def __init__(self, question, iterations=1):
        self.question = question
        self.iterations = iterations

    def data_Generator(self, m, k, eps=1):
        data = []
        if m == 0:
            return data

        for i in range(m * self.iterations):
            xy = []

            # d = d * 1 use in w
            x_0 = 1.0
            xy.append(x_0)

            # x_i for i = 1, ..., k-1
            x_i = list(np.random.randn(1, k-1)[0])
            xy.extend(x_i)

            # x_k = +/- (eps + D)
            x_k = random.choice([-1, 1]) * (eps + np.random.standard_exponential(1)[0])
            xy.append(x_k)
            
            
            y = 1 if x_k > 0 else -1
            xy.append(y)
            data.append(xy)

        filename = 'data/question' + str(self.question) + '/data_k_' + str(k) + '_m_' + str(m) + '_eps_' + str(eps) + '.csv'
        with open(filename, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in data:
                spamwriter.writerow(row)
       
        return data


'''
for test

if __name__ == "__main__":
    dg = DataGenerator()
    print(dg.data_Generator(100, 20, 1))
    print([True if i[-1]*i[-2] > 0 else False for i in dg.data_Generator(100, 20, 1)])
    
'''