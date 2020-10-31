import csv
import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def cal_MOM(data, L, n, j):
    L_bar = 2 * np.mean(data)

    L_true = [L]
    L_pred = [L_bar]
    MSE_bar = mean_squared_error(L_true, L_pred)

    MSE_T = L * L / (3 * n)
    
    return L_bar, MSE_bar, MSE_T


def cal_MLE(data, L, n, j):
    L_bar = max(data)

    L_true = [L]
    L_pred = [L_bar]
    MSE_bar = mean_squared_error(L_true, L_pred)

    MSE_T = L * L * (n - 1) * (n - 2) / (3 * n * (n + 1) * (n + 2))
    
    return L_bar, MSE_bar, MSE_T


if __name__ == "__main__":
    j = 1000
    n = 100
    L = 10
    data = []
    result = []
    result.append(["j", "L_MOM", "L_MLE", "MSE_MOM", "MSE_MLE", "MSE_MOM_T", "MSE_MLE_T"])

    with open('data.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            rows = []
            for i in row:
                rows.append(float(i))
            #print(row)
            data.append(rows)
            #print(', '.join(row))
            #print(data)

    #data_numpy = np.array(data)
    #print(data[0])
    '''
    L_MOM = []
    L_MLE = []
    MSE_MOM = []
    MSE_MLE = []
    MSE_MOM_T = []
    MSE_MLE_T = []
    '''

    for i in range(j):
        L_MOM, MSE_MOM, MSE_MOM_T = cal_MOM(data[i], L, n, j)
        L_MLE, MSE_MLE, MSE_MLE_T = cal_MLE(data[i], L, n, j)
        result.append([i + 1, L_MOM, L_MLE, MSE_MOM, MSE_MLE, MSE_MOM_T, MSE_MLE_T])

    

    with open('result.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in result:
            spamwriter.writerow(row)
        '''
        spamwriter.writerow(L_MOM)
        spamwriter.writerow(L_MLE)
        spamwriter.writerow(MSE_MOM)
        spamwriter.writerow(MSE_MLE)
        spamwriter.writerow(MSE_MOM_T)
        spamwriter.writerow(MSE_MLE_T)
        '''
