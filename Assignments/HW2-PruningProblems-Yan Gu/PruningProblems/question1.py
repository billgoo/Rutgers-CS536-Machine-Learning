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


def test_score(iteration, classifier):
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


def show_Picture(x_data, y_data, x_label, y_label, title):
    plt.figure(figsize=(16, 8))

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.plot(x_data, y_data, marker='.', c='red', lw=0.5)

    plt.legend(loc='upper left')

    filename = 'images/question1/Figure' + title[4] + '.png'

    # save the picture，filename is title
    plt.savefig(filename, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    estimation = []
    M = []
    for i in range(1, 4):
        M.extend(list(range(pow(10, i), 10 * pow(10, i), pow(10, i))))
    M.append(10000)
    typical_err_object = CalTypicalError()
    typical_err_object.data_Generator(40000)
    for m in M:
        # m in list [10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,
        # 2000,3000,4000,5000,6000,7000,8000,9000,10000]
        dg = DataGenerator()
        dg.data_Generator(m)
        fname = 'data/data_m_' + str(m) + '.csv'
        dataset = pd.read_csv(fname, names=['X0','X1','X2','X3','X4','X5',
                                        'X6','X7','X8','X9','X10',
                                        'X11','X12','X13','X14','X15',
                                        'X16','X17','X18','X19','X20','Y'])
        classifier = DecisionTreeClassifier(m, dataset)
        tree = classifier.fit_ID3(classifier.tree_with_data, dataset, dataset, dataset.columns[:-1])
        plotter(m, tree)

        # err_train = classifier.score(dataset)
        err_test = test_score(40000, classifier)
        estimation.append([m, err_test])

    # output the data to be re-format
    with open('data/question1.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in estimation:
            spamwriter.writerow(row)

    # re-format and draw the xy-coordinate figure
    datamap = pd.read_csv('data/question1.csv', names=['m', 'err_f'])
    col_m = datamap['m']
    err_f = datamap['err_f']
    show_Picture(col_m, err_f, "m", "err_f", 
                "Fig 1: Typical error of tree for different value of m.")
                

    # re-format and draw the xy-coordinate figure
    datamap = pd.read_csv('data/question1.csv', names=['m', 'err_f'])
    col_m = datamap['m'][0:9]
    err_f = datamap['err_f'][0:9]
    show_Picture(col_m, err_f, "m", "err_f", 
                "Fig 2: Typical error of tree for different value of m.")

