import pandas as pd
import numpy as np
import math
import csv
import random
import matplotlib.pyplot as plt
from pprint import pprint
from data_generator import DataGenerator
from decision_tree_classifier import DecisionTreeClassifier
from decision_tree_plotter import plotter
from cal_typical_error import CalTypicalError


def test_score(k, iteration, classifier):
    filename = 'data/question5/data_k_' + str(k) + '_iter_' + str(iteration) + '.csv'
    dataset = pd.read_csv(filename, names=['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','Y'])
    # print(dataset)

    print("The text format tree is: ")
    pprint(classifier.tree)
    # pprint(classifier.tree_with_data)

    err_typical = classifier.score(dataset)
        
    return err_typical

'''
def data_Generator(k, iter = 40000):
    csv_xy = []
    data = []
    if k == 0:
        return data

    for i in range(iter):
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

    filename = 'data/data_k_' + str(k) + '_iter_' + str(iter) + '.csv'
    with open(filename, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in csv_xy:
            spamwriter.writerow(row)
       
    return data
'''


def show_Picture(x_data, y_data, x_label, y_label, title):
    plt.figure(figsize=(16, 8))

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.plot(x_data, y_data, c='red', lw=0.5)

    plt.legend(loc='upper left')

    filename = 'images/question6/Figure' + title[4] + '.png'

    # save the picture，filename is title
    plt.savefig(filename, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    estimation = []
    k = 10
    '''
    data_Generator(k)
    '''
    for i in range(1, 52):
        # 20 to 1020 every 20 is one step
        m = i * 20
        '''
        dg = DataGenerator()
        dg.data_Generator(k, m)
        '''
        # use the same data in question5 to get the result
        fname = filename = 'data/question5/data_k_' + str(k) + '_m_' + str(m) + '.csv'
        dataset = pd.read_csv(fname, names=['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','Y'])
        classifier = DecisionTreeClassifier(k, m, dataset)
        tree = classifier.fit_CART(classifier.tree_with_data, dataset, dataset, dataset.columns[:-1])
        plotter(k, m, tree)

        err_train = classifier.score(dataset)
        err_test = test_score(k, 40000, classifier)
        estimation.append([m, err_train, err_test, abs(err_train - err_test)])

    # because k is 10 so specific for m = 2 ^ k = 1024
    # use local variables so scope in a for loop
    for i in range(1024, 1025):
        m = i
        '''
        dg = DataGenerator()
        dg.data_Generator(k, m)
        '''
        fname = filename = 'data/question5/data_k_' + str(k) + '_m_' + str(m) + '.csv'
        dataset = pd.read_csv(fname, names=['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','Y'])
        classifier = DecisionTreeClassifier(k, m, dataset)
        tree = classifier.fit_CART(classifier.tree_with_data, dataset, dataset, dataset.columns[:-1])
        plotter(k, m, tree)

        err_train = classifier.score(dataset)
        err_test = test_score(k, 40000, classifier)
        estimation.append([m, err_train, err_test, abs(err_train - err_test)])

    # output the data to be re-format
    with open('data/question6.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in estimation:
            spamwriter.writerow(row)

    # re-format and draw the xy-coordinate figure
    datamap = pd.read_csv('data/question6.csv', names=['m','err_train','err_test','|err_train - err_test|'])
    col_m = datamap['m']
    gap_between_err = datamap['|err_train - err_test|']
    show_Picture(col_m, gap_between_err, "m", "|err_train - err_test|", 
                "Fig 1: |err_train - err_test| for different value of m.")

