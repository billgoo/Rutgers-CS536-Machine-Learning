import pandas as pd
import numpy as np
import math
import csv
import random
import matplotlib.pyplot as plt
from pprint import pprint
from decision_tree_classifier import DecisionTreeClassifier
from decision_tree_plotter import plotter
from collections import Counter


def show_Picture(x_data, y_data_1, y_data_2, y_data_name1, y_data_name2, x_label, y_label, title):
    plt.figure(figsize=(16, 8))

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.plot(x_data, y_data_1, marker='.', c='red', lw=0.5, label=y_data_name1)
    plt.plot(x_data, y_data_2, marker='x', c='blue', lw=0.5, label=y_data_name2)
    
    for i in range(len(x_data)):
        plt.annotate("%.4f" % y_data_1[i], xy=(x_data[i],y_data_1[i]), xytext=(-20, -20), textcoords='offset points', color='red')
        plt.annotate("%.4f" % y_data_2[i], xy=(x_data[i],y_data_2[i]), xytext=(-20, 13), textcoords='offset points', color='blue')

    plt.legend(loc='upper right')

    filename = 'images/question3/Figure.' + title[4:6] + '.png'

    # save the picture，filename is title
    plt.savefig(filename, bbox_inches='tight')

    plt.show()



if __name__ == "__main__":
    estimation = []
    m = 10000
    fname = 'data/question3/data_m_' + str(m) + '.csv'
    dataset = pd.read_csv(fname, names=['X0','X1','X2','X3','X4','X5',
                                        'X6','X7','X8','X9','X10',
                                        'X11','X12','X13','X14','X15',
                                        'X16','X17','X18','X19','X20','Y'])
    train_data = dataset[0:8000] # 0-7999, totally 8000 data
    test_data = dataset[8000:].reset_index(drop=True) # 8000-9999, totally 2000 data
    for d in range(21):
        classifier = DecisionTreeClassifier(m, train_data)
        tree = classifier.fit_ID3_Pruning_Depth(d, classifier.tree_with_data, train_data, 
                                                train_data, train_data.columns[:-1])
                                                
        # error for train and test data
        err_train = classifier.score_Pruning(train_data, tree)
        err_test = classifier.score_Pruning(test_data, tree)

        estimation.append([d, err_train, err_test])
        print(d, err_train, err_test)
        print("Tree with depth = " + str(d) + " is :")
        pprint(tree)
    
    # output the data to be re-format
    with open('data/question3/3a.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in estimation:
            spamwriter.writerow(row)

    # re-format and draw the xy-coordinate figure
    datamap = pd.read_csv('data/question3/3a.csv', names=['depth','err_train','err_test'])
    col_d = datamap['depth']
    col_train = datamap['err_train']
    col_test = datamap['err_test']
    show_Picture(col_d, col_train, col_test, "err_train", "err_test", 
                "Depth", "Error of the tree", "Fig 3a: Error of decision tree for different depth.")
    