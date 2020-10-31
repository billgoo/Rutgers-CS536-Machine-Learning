import csv
import random
import numpy as np
import math
from decision_tree_classifier import DecisionTreeClassifier
import pandas as pd
from pprint import pprint

class CalTypicalError:

    def __init__(self):
        pass


    def test_score(self, k, m, iteration, classifier):
        self.data_Generator(k, m, iteration)
        filename = 'data/data_k_' + str(k) + '_m_' + str(m) + '_iter_' + str(iteration) + '.csv'
        dataset = pd.read_csv(filename, names=['X1','X2','X3','X4','Y'])
        # print(dataset)

        print("The text format tree is: ")
        pprint(classifier.tree)
        # pprint(classifier.tree_with_data)

        err_typical = classifier.score(dataset)
        
        return err_typical


    def data_Generator(self, k, m, iter = 1):
        csv_xy = []
        data = []
        if m == 0 or k == 0:
            return data

        for i in range(m * iter):
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

        filename = 'data/data_k_' + str(k) + '_m_' + str(m) + '_iter_' + str(iter) + '.csv'
        with open(filename, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in csv_xy:
                spamwriter.writerow(row)
       
        return data



if __name__ == "__main__":
    pass