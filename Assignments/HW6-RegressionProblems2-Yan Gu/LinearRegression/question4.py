import pandas as pd
import math
import matplotlib.pyplot as plt
import csv
import numpy as np
from data_generator import DataGenerator
from lassoRegression import LassoRegression


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


if __name__ == "__main__":
    '''
    lambd_set1 = [0,5,10,15,20,30,40,50,60,70,80]
    lambd_set2 = [100,120,140,160,180,200]
    lambd_set3 = [400,600,800]
    
    result = []
    for lambd in lambd_set1:
        print(lambd)
        data = pd.read_csv('LinearRegression/data/question3/results3.1.csv', 
                        names=['lambd','b','X1','X2','X3','X4','X5',
                            'X6','X7','X8','X9','X10',
                            'X11','X12','X13','X14','X15',
                            'X16','X17','X18','X19','X20','train_err','zero_count'])
        weight = weight = data[data['lambd']==lambd][data.columns[1:22]].values

        filename1 = 'LinearRegression/data/question1_m_1000000.csv'
        test_data = pd.read_csv(filename1, names=['b','X1','X2','X3','X4','X5',
                                                'X6','X7','X8','X9','X10',
                                                'X11','X12','X13','X14','X15',
                                                'X16','X17','X18','X19','X20','y'])
        test_X, test_y = test_data[test_data.columns[:-1]].values, test_data[['y']].values
        test_err = 0.0
        for i in range(1000000):
            test_err += math.pow((np.dot(test_X[i], weight.T) - test_y[i]), 2)
        test_err /= 1000000
        result.append([lambd, test_err])
        print([lambd, test_err])
    
    for lambd in lambd_set2:
        print(lambd)
        data = pd.read_csv('LinearRegression/data/question3/results3.2.csv', 
                        names=['lambd','b','X1','X2','X3','X4','X5',
                            'X6','X7','X8','X9','X10',
                            'X11','X12','X13','X14','X15',
                            'X16','X17','X18','X19','X20','train_err','zero_count'])
        weight = weight = data[data['lambd']==lambd][data.columns[1:22]].values

        filename1 = 'LinearRegression/data/question1_m_1000000.csv'
        test_data = pd.read_csv(filename1, names=['b','X1','X2','X3','X4','X5',
                                                'X6','X7','X8','X9','X10',
                                                'X11','X12','X13','X14','X15',
                                                'X16','X17','X18','X19','X20','y'])
        test_X, test_y = test_data[test_data.columns[:-1]].values, test_data[['y']].values
        test_err = 0.0
        for i in range(1000000):
            test_err += math.pow((np.dot(test_X[i], weight.T) - test_y[i]), 2)
        test_err /= 1000000
        result.append([lambd, test_err])
        print([lambd, test_err])

    for lambd in lambd_set3:
        print(lambd)
        data = pd.read_csv('LinearRegression/data/question3/results3.3.csv', 
                        names=['lambd','b','X1','X2','X3','X4','X5',
                            'X6','X7','X8','X9','X10',
                            'X11','X12','X13','X14','X15',
                            'X16','X17','X18','X19','X20','train_err','zero_count'])
        weight = weight = data[data['lambd']==lambd][data.columns[1:22]].values

        filename1 = 'LinearRegression/data/question1_m_1000000.csv'
        test_data = pd.read_csv(filename1, names=['b','X1','X2','X3','X4','X5',
                                                'X6','X7','X8','X9','X10',
                                                'X11','X12','X13','X14','X15',
                                                'X16','X17','X18','X19','X20','y'])
        test_X, test_y = test_data[test_data.columns[:-1]].values, test_data[['y']].values
        test_err = 0.0
        for i in range(1000000):
            test_err += math.pow((np.dot(test_X[i], weight.T) - test_y[i]), 2)
        test_err /= 1000000
        result.append([lambd, test_err])
        print([lambd, test_err])

    # output the data to be re-format
    with open('LinearRegression/data/question4/results.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in result:
            spamwriter.writerow(row)

    # re-format and draw the xy-coordinate figure
    datamap = pd.read_csv('LinearRegression/data/question4/results.csv', 
                        names=['lambd','test_error'])
    col_l = datamap['lambd']
    col_e = datamap['test_error']
    show_Picture(col_l, col_e, "True error", "lambd", "Testing error for each lambd", 
                "Fig 4: True error as function of lambd.")
      
    result = []
    for lambd in range(0, 81, 5):
        print(lambd)
        data = pd.read_csv('LinearRegression/data/question3/results3.1.csv', 
                        names=['lambd','b','X1','X2','X3','X4','X5',
                            'X6','X7','X8','X9','X10',
                            'X11','X12','X13','X14','X15',
                            'X16','X17','X18','X19','X20','train_err','zero_count'])
        weight = weight = data[data['lambd']==lambd][data.columns[1:22]].values

        filename1 = 'LinearRegression/data/question1_m_1000000.csv'
        test_data = pd.read_csv(filename1, names=['b','X1','X2','X3','X4','X5',
                                                'X6','X7','X8','X9','X10',
                                                'X11','X12','X13','X14','X15',
                                                'X16','X17','X18','X19','X20','y'])
        test_X, test_y = test_data[test_data.columns[:-1]].values, test_data[['y']].values
        test_err = 0.0
        for i in range(1000000):
            test_err += math.pow((np.dot(test_X[i], weight.T) - test_y[i]), 2)
        test_err /= 1000000
        result.append([lambd, test_err])
        print([lambd, test_err])
    
    # output the data to be re-format
    with open('LinearRegression/data/question4/results4.2.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in result:
            spamwriter.writerow(row)

    # re-format and draw the xy-coordinate figure
    datamap = pd.read_csv('LinearRegression/data/question4/results4.2.csv', 
                        names=['lambd','test_error'])
    col_l = datamap['lambd']
    col_e = datamap['test_error']
    show_Picture(col_l, col_e, "True error", "lambd", "Testing error for each lambd", 
                "Fig 4: True error as function of lambd.")
    '''  
    result = []
    for lambd in range(35):
        print(lambd)
        data = pd.read_csv('LinearRegression/data/question3/results3.1.csv', 
                        names=['lambd','b','X1','X2','X3','X4','X5',
                            'X6','X7','X8','X9','X10',
                            'X11','X12','X13','X14','X15',
                            'X16','X17','X18','X19','X20','train_err','zero_count'])
        weight = weight = data[data['lambd']==lambd][data.columns[1:22]].values

        filename1 = 'LinearRegression/data/question1_m_1000000.csv'
        test_data = pd.read_csv(filename1, names=['b','X1','X2','X3','X4','X5',
                                                'X6','X7','X8','X9','X10',
                                                'X11','X12','X13','X14','X15',
                                                'X16','X17','X18','X19','X20','y'])
        test_X, test_y = test_data[test_data.columns[:-1]].values, test_data[['y']].values
        test_err = 0.0
        for i in range(1000000):
            test_err += math.pow((np.dot(test_X[i], weight.T) - test_y[i]), 2)
        test_err /= 1000000
        result.append([lambd, test_err])
        print([lambd, test_err])
    
    # output the data to be re-format
    with open('LinearRegression/data/question4/results4.3.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in result:
            spamwriter.writerow(row)

    # re-format and draw the xy-coordinate figure
    datamap = pd.read_csv('LinearRegression/data/question4/results4.3.csv', 
                        names=['lambd','test_error'])
    col_l = datamap['lambd']
    col_e = datamap['test_error']
    show_Picture(col_l, col_e, "True error", "lambd", "Testing error for each lambd", 
                "Fig 4: True error as function of lambd.")