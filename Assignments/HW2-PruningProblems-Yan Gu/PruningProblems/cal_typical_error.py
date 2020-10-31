import csv
import random
import numpy as np
import math
from decision_tree_classifier import DecisionTreeClassifier
import pandas as pd
from pprint import pprint
from collections import Counter

class CalTypicalError:

    def __init__(self):
        pass


    def test_score(self, iteration, classifier):
        # self.data_Generator(iteration)
        filename = 'data/data_iter_' + str(iteration) + '.csv'
        dataset = pd.read_csv(filename, names=['X0','X1','X2','X3','X4','X5',
                                            'X6','X7','X8','X9','X10',
                                            'X11','X12','X13','X14','X15',
                                            'X16','X17','X18','X19','X20','Y'])
        # print(dataset)

        print("The text format tree is: ")
        pprint(classifier.tree)
        # pprint(classifier.tree_with_data)

        err_typical = classifier.score(dataset)
        
        return err_typical


    def data_Generator(self, iter = 1):
        csv_xy = []
        data = []

        for i in range(iter):
            x = []
            csv_x = []
            x_0 = random.choice([0,1]) # i = 0
            x.append(x_0)
            csv_x.append(x_0)
            for j in range(1, 15):
                # i = 1 - 14
                x_i_1 = x[-1]
                if x_i_1:
                    x_i = np.random.multinomial(1, [.25, .75]).tolist().index(1)
                else:
                    x_i = np.random.multinomial(1, [.75, .25]).tolist().index(1)
                x.append(x_i)
                csv_x.append(x_i)
                
            for j in range(15, 21):
                # i = 15 - 20
                x_i = random.choice([0,1])
                x.append(x_i)
                csv_x.append(x_i)
            
            # majority[i] is where i = x0 and use Counter to 
            # count the majority 0 or 1 in the following domain
            majority = [Counter(x[1:8]).most_common(1)[0][0], Counter(x[8:15]).most_common(1)[0][0]]
            # Y = majority[x[0]]
            data.append([x, majority[x[0]]])
            csv_x.append(majority[x[0]])

            csv_xy.append(csv_x)

        filename = 'data/data_iter_' + str(iter) + '.csv'
        with open(filename, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in csv_xy:
                spamwriter.writerow(row)
       
        return data



if __name__ == "__main__":
    pass