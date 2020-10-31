import pandas as pd
import math
import matplotlib.pyplot as plt
import csv
import numpy as np
from data_generator import DataGenerator
from  lassoRegression import LassoRegression


def show_Picture(x_data, y_data, y_data_name, x_label, y_label, title):
    plt.figure(figsize=(16, 8))

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.plot(x_data, y_data, marker='.', c='red', lw=0.5, label=y_data_name)

    filename = 'LinearRegression/images/Figure.' + title[4] + '.png'

    # save the picture，filename is title
    plt.savefig(filename, bbox_inches='tight')

    plt.show()


def plotLambda(lambd=0.0):
    filename = 'LinearRegression/data/question1_m_1000.csv'
    train_data = pd.read_csv(filename, names=['b','X1','X2','X3','X4','X5',
                                            'X6','X7','X8','X9','X10',
                                            'X11','X12','X13','X14','X15',
                                            'X16','X17','X18','X19','X20','y'])

    # print(train_data)
    train_X, train_y = train_data[train_data.columns[:-1]].values, train_data[['y']].values
    #print(train_X.shape, train_y.shape)
    # train
    model = LassoRegression(lambd)
    model.fit(np.matrix(train_X), np.matrix(train_y))

    zero_count = 0
    result = []
    result.append(lambd)
    for i in model.param:
        result.append(i[0])
        if i[0] == 0.0:
            zero_count += 1

    result.append(model.train_err)
    result.append(zero_count)

    return result




if __name__ == "__main__":
    '''
    result = []
    for i in range(81):
        lambd = i
        result.append(plotLambda(lambd))

    # output the data to be re-format
    with open('LinearRegression/data/question3/results3.1.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in result:
            spamwriter.writerow(row)

    # re-format and draw the xy-coordinate figure
    datamap = pd.read_csv('LinearRegression/data/question3/results3.1.csv', 
                        names=['lambd','b','X1','X2','X3','X4','X5',
                            'X6','X7','X8','X9','X10',
                            'X11','X12','X13','X14','X15',
                            'X16','X17','X18','X19','X20','train_err','zero_count'])
    col_l = datamap['lambd']
    col_e = datamap['zero_count']
    show_Picture(col_l, col_e, "Number", "lambd", "Number of features eliminated for each lambd", 
                "Fig 3: Number of features been eliminated as function of lambd.")
    
    result = []
    for i in range(80, 201, 20):
        lambd = i
        result.append(plotLambda(lambd))

    # output the data to be re-format
    with open('LinearRegression/data/question3/results3.2.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in result:
            spamwriter.writerow(row)

    # re-format and draw the xy-coordinate figure
    datamap = pd.read_csv('LinearRegression/data/question3/results3.2.csv', 
                        names=['lambd','b','X1','X2','X3','X4','X5',
                            'X6','X7','X8','X9','X10',
                            'X11','X12','X13','X14','X15',
                            'X16','X17','X18','X19','X20','train_err','zero_count'])
    col_l = datamap['lambd']
    col_e = datamap['zero_count']
    show_Picture(col_l, col_e, "Number", "lambd", "Number of features eliminated for each lambd", 
                "Fig 3: Number of features been eliminated as function of lambd.")
    '''
    result = []
    for i in range(3, 9):
        lambd = i * 100
        result.append(plotLambda(lambd))

    # output the data to be re-format
    with open('LinearRegression/data/question3/results3.3.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in result:
            spamwriter.writerow(row)

    # re-format and draw the xy-coordinate figure
    datamap = pd.read_csv('LinearRegression/data/question3/results3.3.csv', 
                        names=['lambd','b','X1','X2','X3','X4','X5',
                            'X6','X7','X8','X9','X10',
                            'X11','X12','X13','X14','X15',
                            'X16','X17','X18','X19','X20','train_err','zero_count'])
    col_l = datamap['lambd']
    col_e = datamap['zero_count']
    show_Picture(col_l, col_e, "Number", "lambd", "Number of features eliminated for each lambd", 
                "Fig 3: Number of features been eliminated as function of lambd.")
                