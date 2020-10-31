import pandas as pd
import numpy as np
import math
import csv
from pprint import pprint
from decision_tree_classifier import DecisionTreeClassifier


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
    sample_size = 156
    m = 10000
    num_noise = dict()
    total_num = 0.0
    total_num_p = 0.0
    for iter in range(1, 201):
        fname = 'data/question567/data_m_' + str(m) + '_iter_' + str(iter) + '.csv'
        dataset = pd.read_csv(fname, names=['X0','X1','X2','X3','X4','X5',
                                            'X6','X7','X8','X9','X10',
                                            'X11','X12','X13','X14','X15',
                                            'X16','X17','X18','X19','X20','Y'])
        classifier = DecisionTreeClassifier(m, dataset)
        tree = classifier.fit_ID3(classifier.tree_with_data, dataset, dataset, dataset.columns[:-1])
        classifier_p = DecisionTreeClassifier(m, dataset)
        tree_p = classifier_p.fit_ID3_Pruning_Size(sample_size, classifier_p.tree_with_data, dataset, dataset, 
                                                    dataset.columns[:-1])
        total_num += find_Noise_Node(tree)
        total_num_p += find_Noise_Node(tree_p)
        # pprint(tree)
        print("Loop: iter = " + str(iter))
    num_noise['origin tree'] = total_num / 200.0
    num_noise['pruning tree'] = total_num_p / 200.0
    
    print(num_noise)
    