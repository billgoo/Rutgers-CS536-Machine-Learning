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

            # x_i for i = 1, ..., k
            x_i = list(np.random.randn(1, k)[0])
            xy.extend(x_i)            
            
            sum_x = sum(pow(i, 2) for i in x_i)
            y = 1 if sum_x >= k else -1
            xy.append(y)
            data.append(xy)

        filename = 'data/question' + str(self.question) + '/data_k_' + str(k) + '_m_' + str(m) + '_eps_' + str(eps) + '.csv'
        with open(filename, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in data:
                spamwriter.writerow(row)
       
        return data



if __name__ == "__main__":
    dg = DataGenerator(5)
    print(dg.data_Generator(100, 2, 1))
    # if linear separable then all will be true
    # print([True if (sum(pow(j, 2) for j in i[1:-1])-2)*i[-1] >= 0 else False for i in dg.data_Generator(100, 2, 1)])
    
