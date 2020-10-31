import pandas as pd
import math
import matplotlib.pyplot as plt
import csv
from data_generator import DataGenerator
from ridgeRegression import RidgeRegression


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
    dataset = pd.read_csv(filename, names=['b','X1','X2','X3','X4','X5',
                                            'X6','X7','X8','X9','X10',
                                            'X11','X12','X13','X14','X15',
                                            'X16','X17','X18','X19','X20','y'])

    train_data = dataset.drop(['X7','X14','X16','X17','X18','X19','X20'], axis=1)

    # print(train_data)
    train_X, train_y = train_data[train_data.columns[:-1]].values, train_data[['y']].values
    #print(train_X.shape, train_y.shape)
    # train
    model = RidgeRegression(lambd)
    model.fit(train_X, train_y)

    train_err = 0.0
    for i in range(1000):
        train_err += math.pow((model.predict(train_X[i])-train_y[i]), 2)
    train_err /= 1000

    result = []
    result.append(lambd)
    for i in model.param:
        result.append(i[0])
    result.append(train_err)

    filename1 = 'LinearRegression/data/question1_m_1000000.csv'
    testset = pd.read_csv(filename1, names=['b','X1','X2','X3','X4','X5',
                                            'X6','X7','X8','X9','X10',
                                            'X11','X12','X13','X14','X15',
                                            'X16','X17','X18','X19','X20','y'])
    test_data = testset.drop(['X7','X14','X16','X17','X18','X19','X20'], axis=1)
    test_X, test_y = test_data[test_data.columns[:-1]].values, test_data[['y']].values
    test_err = 0.0
    for i in range(1000000):
        test_err += math.pow((model.predict(test_X[i])-test_y[i]), 2)
    test_err /= 1000000

    result.append(test_err)
    return result




if __name__ == "__main__":
    '''
    result = []
    for i in range(21):
        lambd = 0.2 * i
        result.append(plotLambda(lambd))

    # output the data to be re-format
    with open('LinearRegression/data/question5/results5.1.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in result:
            spamwriter.writerow(row)

    # re-format and draw the xy-coordinate figure
    datamap = pd.read_csv('LinearRegression/data/question5/results5.1.csv', 
                        names=['lambd','b','X1','X2','X3','X4','X5',
                            'X6','X8','X9','X10',
                            'X11','X12','X13','X15',
                            'train_err','test_err'])
    col_l = datamap['lambd']
    col_e = datamap['test_err']
    show_Picture(col_l, col_e, "True error", "lambd", "Testing error for each lambd", 
                "Fig 5: True error as function of lambd.")
    '''
    result = []
    for i in range(11):
        lambd = 0.02 * i
        result.append(plotLambda(lambd))

    # output the data to be re-format
    with open('LinearRegression/data/question5/results5.2.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in result:
            spamwriter.writerow(row)

    # re-format and draw the xy-coordinate figure
    datamap = pd.read_csv('LinearRegression/data/question5/results5.2.csv', 
                        names=['lambd','b','X1','X2','X3','X4','X5',
                            'X6','X8','X9','X10',
                            'X11','X12','X13','X15',
                            'train_err','test_err'])
    col_l = datamap['lambd']
    col_e = datamap['test_err']
    show_Picture(col_l, col_e, "True error", "lambd", "Testing error for each lambd", 
                "Fig 5: True error as function of lambd.")
                
                