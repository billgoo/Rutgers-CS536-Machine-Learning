import csv
import random
import numpy as np
import math
from collections import Counter

class DataGenerator:

    def __init__(self, question=-1, iterations=1):
        self.question = question
        self.iterations = iterations

    def data_Generator(self, m):
        data = []
        if m == 0:
            return data

        for i in range(m):
            xy = [1]

            # x 1 - x 10, x 16 - x 20
            x_standard_n = np.random.standard_normal(15)
            for j in x_standard_n:
                xy.append(j)

            x_11 = x_standard_n[0] + x_standard_n[1] + np.random.normal(0, math.sqrt(0.1))
            x_12 = x_standard_n[2] + x_standard_n[3] + np.random.normal(0, math.sqrt(0.1))
            x_13 = x_standard_n[3] + x_standard_n[4] + np.random.normal(0, math.sqrt(0.1))
            x_14 = 0.1 * x_standard_n[6] + np.random.normal(0, math.sqrt(0.1))
            x_15 = 2 * x_standard_n[1] - 10 + np.random.normal(0, math.sqrt(0.1))

            xy.insert(11, x_11)
            xy.insert(12, x_12)
            xy.insert(13, x_13)
            xy.insert(14, x_14)
            xy.insert(15, x_15)
            
            y = 10 + sum(xy[k] * pow(0.6, k+1) for k in range(0, 10)) + np.random.normal(0, math.sqrt(0.1))
            xy.append(y)
            
            data.append(xy)

        filename = 'LinearRegression/data/question' + str(self.question) + '_m_' + str(m) + '.csv'
        with open(filename, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in data:
                spamwriter.writerow(row)
       
        return data


'''
for test

if __name__ == "__main__":
    dg = DataGenerator(1)
    print(dg.data_Generator(1000))

'''