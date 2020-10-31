import csv
import random
import numpy as np
import math

class DataGenerator:

    def __init__(self):
        pass

    def data_Generator(self, k, m, iter = 1):
        csv_xy = []
        data = []
        if m == 0 or k == 0:
            return data

        for i in range(m):
            x = []
            csv_x = []
            x_1 = random.choice([0,1])
            x.append(x_1)
            csv_x.append(x_1)
            w_deno = 0.0 # denominator of w
            prob = 0.0
            for j in range(1, k):
                x_i_1 = x[-1]
                if x_i_1:
                    x_i = np.random.multinomial(1, [.25, .75]).tolist().index(1)
                else:
                    x_i = np.random.multinomial(1, [.75, .25]).tolist().index(1)
                x.append(x_i)
                csv_x.append(x_i)
                w_deno += math.pow(0.9, j + 1)
        
            for j in range(1, k):
                w_i = math.pow(0.9, j + 1) / w_deno
                prob += w_i * x[j]
        
            if prob >= .5:
                data.append([x, x[0]])
                csv_x.append(x[0])     
            else:
                data.append([x, 1 - x[0]])
                csv_x.append(1 - x[0])
            csv_xy.append(csv_x)

        filename = 'data/data_k_' + str(k) + '_m_' + str(m) + '.csv'
        with open(filename, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in csv_xy:
                spamwriter.writerow(row)
       
        return data



if __name__ == "__main__":
    dg = DataGenerator()
    print(dg.data_Generator(4, 30))