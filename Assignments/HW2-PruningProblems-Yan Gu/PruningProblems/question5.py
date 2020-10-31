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


def data_Generator(m, iter = 1):
    csv_xy = []
    data = []
    if m == 0:
        return data

    for i in range(m):
        x = []
        csv_x = []
        x_0 = random.choice([0,1]) # i = 0
        x.append(x_0)
        csv_x.append(x_0)
        for j in range(1, 15):
            # i = 1 - 14
            x_i_1 = x[-1]
            if x_i_1:
                x_i = np.random.multinomial(1, [.25, .75]).tolist().index(1)
            else:
                x_i = np.random.multinomial(1, [.75, .25]).tolist().index(1)
            x.append(x_i)
            csv_x.append(x_i)
                
        for j in range(15, 21):
            # i = 15 - 20
            x_i = random.choice([0,1])
            x.append(x_i)
            csv_x.append(x_i)
            
        # majority[i] is where i = x0 and use Counter to 
        # count the majority 0 or 1 in the following domain
        majority = [Counter(x[1:8]).most_common(1)[0][0], Counter(x[8:15]).most_common(1)[0][0]]
        # Y = majority[x[0]]
        data.append([x, majority[x[0]]])
        csv_x.append(majority[x[0]])

        csv_xy.append(csv_x)

    filename = 'data/question567/data_m_' + str(m) + '_iter_' + str(iter) + '.csv'
    with open(filename, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in csv_xy:
            spamwriter.writerow(row)
       
    return data


def find_Noise_Node(tree):
    features = list(tree.keys())
    total_num = 0.0
    keyword = ['X15','X16','X17','X18','X19','X20']
    for key in features:
        if key in keyword:
            total_num += 1.0
            # print(key)
        if isinstance(tree[key][0], dict):
            # feature variables = 0
            total_num += find_Noise_Node(tree[key][0])
        if isinstance(tree[key][1], dict):
            # feature variables = 1
            total_num += find_Noise_Node(tree[key][1])

    return total_num



if __name__ == "__main__":
    estimation = []
    d = 6
    m = 10000
    num_noise = dict()
    total_num = 0.0
    total_num_p = 0.0
    for iter in range(1, 201):
        # data_Generator(m, iter)
        fname = 'data/question567/data_m_' + str(m) + '_iter_' + str(iter) + '.csv'
        dataset = pd.read_csv(fname, names=['X0','X1','X2','X3','X4','X5',
                                            'X6','X7','X8','X9','X10',
                                            'X11','X12','X13','X14','X15',
                                            'X16','X17','X18','X19','X20','Y'])
        classifier = DecisionTreeClassifier(m, dataset)
        tree = classifier.fit_ID3(classifier.tree_with_data, dataset, dataset, dataset.columns[:-1])
        classifier_p = DecisionTreeClassifier(m, dataset)
        tree_p = classifier_p.fit_ID3_Pruning_Depth(d, classifier_p.tree_with_data, dataset, dataset, 
                                                    dataset.columns[:-1])
        total_num += find_Noise_Node(tree)
        total_num_p += find_Noise_Node(tree_p)
        # pprint(tree)
        print("Loop: iter = " + str(iter))
    num_noise['origin tree'] = total_num / 200.0
    num_noise['pruning tree'] = total_num_p / 200.0
    
    print(num_noise)
    