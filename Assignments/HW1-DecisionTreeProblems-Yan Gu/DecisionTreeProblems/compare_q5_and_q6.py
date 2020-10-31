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

def show_Picture(x_data, y_data_1, y_data_2, y_data_name1, y_data_name2, x_label, y_label, title):
    plt.figure(figsize=(16, 8))

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.plot(x_data, y_data_1, c='red', lw=0.5, label=y_data_name1)
    plt.plot(x_data, y_data_2, c='blue', lw=0.5, label=y_data_name2)

    plt.legend(loc='upper left')

    filename = 'images/Figure.' + title[4] + title[6:17] + '.png'

    # save the picture，filename is title
    plt.savefig(filename, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    # re-format and draw the xy-coordinate figure
    datamap1 = pd.read_csv('data/question5.csv', names=['m','err_train','err_test','|err_train - err_test|'])
    datamap2 = pd.read_csv('data/question6.csv', names=['m','err_train','err_test','|err_train - err_test|'])
    col_m = datamap1['m']
    gap_between_err1 = datamap1['|err_train - err_test|']
    gap_between_err2 = datamap2['|err_train - err_test|']
    '''
    show_Picture(col_m, gap_between_err, "m", "|err_train - err_test|", 
                "Fig 1: |err_train - err_test| for different value of m.")
    '''
    show_Picture(col_m, gap_between_err1, gap_between_err2, "question5", "question6", 
                "m", "|err_train - err_test|", "Fig 2: Comparison of 5 and 6.")