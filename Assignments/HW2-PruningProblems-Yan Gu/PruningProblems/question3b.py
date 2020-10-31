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
        plt.annotate("(%s, %.4f)" % (x_data[i], y_data_1[i]), 
                    xy=(x_data[i],y_data_1[i]), xytext=(-20, -20), textcoords='offset points', color='red')
        plt.annotate("(%s, %.4f)" % (x_data[i], y_data_1[i]), 
                    xy=(x_data[i],y_data_2[i]), xytext=(-20, 13), textcoords='offset points', color='blue')

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
    x_points = [1]
    i = 10000
    while i:
        x_points.append(i)
        i = i // 2
    # we have already know 2^13 < 10000 < 2^14
    # so len(S) = 13
    # to increase the sample size in S, we choose every size to be the average of
    # each two numbers in the S
    for j in range(1, 13):
        x_points.append((x_points[j-1] + x_points[j]) // 2)
    S = list(set(x_points))
    S.sort()

    for s in S:
        classifier = DecisionTreeClassifier(m, train_data)
        tree = classifier.fit_ID3_Pruning_Size(s, classifier.tree_with_data, train_data, 
                                                train_data, train_data.columns[:-1])
                                                
        # error for train and test data
        err_train = classifier.score_Pruning(train_data, tree)
        err_test = classifier.score_Pruning(test_data, tree)

        estimation.append([s, err_train, err_test])
        print(s, err_train, err_test)
        print("Tree with sample size = " + str(s) + " is :")
        pprint(tree)
    
    # output the data to be re-format
    with open('data/question3/3b.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in estimation:
            spamwriter.writerow(row)

    # re-format and draw the xy-coordinate figure
    datamap = pd.read_csv('data/question3/3b.csv', names=['minmum sample size','err_train','err_test'])
    col_d = datamap['minmum sample size']
    col_train = datamap['err_train']
    col_test = datamap['err_test']
    show_Picture(col_d, col_train, col_test, "err_train", "err_test", 
                "Minmum sample size", "Error of the tree", "Fig 3b: Error of decision tree for different minmum sample size.")
    show_Picture(col_d[0:9].reset_index(drop=True), col_train[0:9].reset_index(drop=True), 
                col_test[0:9].reset_index(drop=True), 
                "err_train", "err_test", "Minmum sample size", "Error of the tree", 
                "Fig 3b(1): Error of decision tree for different minmum sample size.")
    show_Picture(col_d[6:14].reset_index(drop=True), col_train[6:14].reset_index(drop=True), 
                col_test[6:14].reset_index(drop=True), 
                "err_train", "err_test", "Minmum sample size", "Error of the tree", 
                "Fig 3b(2): Error of decision tree for different minmum sample size.")
    show_Picture(col_d[11:19].reset_index(drop=True), col_train[11:19].reset_index(drop=True), 
                col_test[11:19].reset_index(drop=True), 
                "err_train", "err_test", "Minmum sample size", 
                "Error of the tree", "Fig 3b(3): Error of decision tree for different minmum sample size.")
    show_Picture(col_d[16:22].reset_index(drop=True), col_train[16:22].reset_index(drop=True), 
                col_test[16:22].reset_index(drop=True), 
                "err_train", "err_test", "Minmum sample size", "Error of the tree", 
                "Fig 3b(4): Error of decision tree for different minmum sample size.")
    