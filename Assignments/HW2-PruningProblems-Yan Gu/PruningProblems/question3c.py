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
        plt.annotate("(%.3f, %.4f)" % (x_data[i], y_data_1[i]), 
                    xy=(x_data[i],y_data_1[i]), xytext=(-20, -20), textcoords='offset points', color='red')
        plt.annotate("(%.3f, %.4f)" % (x_data[i], y_data_1[i]), 
                    xy=(x_data[i],y_data_2[i]), xytext=(-20, 13), textcoords='offset points', color='blue')

    plt.legend(loc='upper right')

    filename = 'images/question3/Figure.' + title[4:9] + '.png'

    # save the picture，filename is title
    plt.savefig(filename, bbox_inches='tight')

    plt.show()



if __name__ == "__main__":
    estimation = []
    P_T_0 = [0.50, 0.40, 0.25, 0.15, 0.10, 0.05, 0.025, 0.010, 0.005, 0.001] # Probability of Independent
    T_0 = [0.455, 0.708, 1.323, 2.072, 2.706, 3.841, 5.024, 6.635, 7.879, 10.828]
    m = 10000
    fname = 'data/question3/data_m_' + str(m) + '.csv'
    dataset = pd.read_csv(fname, names=['X0','X1','X2','X3','X4','X5',
                                        'X6','X7','X8','X9','X10',
                                        'X11','X12','X13','X14','X15',
                                        'X16','X17','X18','X19','X20','Y'])
    train_data = dataset[0:8000] # 0-7999, totally 8000 data
    test_data = dataset[8000:].reset_index(drop=True) # 8000-9999, totally 2000 data
    for i in range(len(T_0)):
        classifier = DecisionTreeClassifier(m, train_data)
        tree = classifier.fit_ID3_Pruning_Sig(T_0[i], classifier.tree_with_data, train_data, 
                                                train_data, train_data.columns[:-1])
                                                
        # error for train and test data
        err_train = classifier.score_Pruning(train_data, tree)
        err_test = classifier.score_Pruning(test_data, tree)

        estimation.append([P_T_0[i], T_0[i], err_train, err_test])
        print(T_0[i], err_train, err_test)
        print("Tree with T0 = " + str(T_0[i]) + " is :")
        pprint(tree)
    
    # output the data to be re-format
    with open('data/question3/3c.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in estimation:
            spamwriter.writerow(row)

    # re-format and draw the xy-coordinate figure
    datamap = pd.read_csv('data/question3/3c.csv', names=['Probability','T0','err_train','err_test'])
    col_t = datamap['T0']
    col_p = datamap['Probability']
    col_train = datamap['err_train']
    col_test = datamap['err_test']
    show_Picture(col_t, col_train, col_test, "err_train", "err_test", 
                "T0", "Error of the tree", "Fig 3c(1): Error of decision tree for different T0.")
    show_Picture(col_p, col_train, col_test, "err_train", "err_test", 
                "Probability of independent", "Error of the tree", 
                "Fig 3c(2): Error of decision tree for different probability of independent.")
    