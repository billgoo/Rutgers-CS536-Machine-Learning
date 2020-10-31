import pandas as pd
import numpy as np
from pprint import pprint
from data_generator import DataGenerator
from decision_tree_classifier import DecisionTreeClassifier
from decision_tree_plotter import plotter
from cal_typical_error import CalTypicalError

if __name__ == "__main__":

    '''
    general data
    dg = DataGenerator()
    dg.data_Generator(4, 30)
    '''

    dataset = pd.read_csv('data/data_k_4_m_30.csv',
                        names=['X1','X2','X3','X4','Y'])
    # print(dataset)

    classifier = DecisionTreeClassifier(4, 30, dataset)
    tree = classifier.fit_ID3(classifier.tree_with_data, dataset, dataset, dataset.columns[:-1])
    print("The text format tree is: ")
    pprint(tree)
    # pprint(classifier.tree_with_data)

    err_train = classifier.score(dataset)
    print("err_train = ", err_train)

    '''
    typical_err = CalTypicalError()
    err_test = typical_err.test_score(4, 30, 1000, classifier)
    '''
    typical_err = CalTypicalError()
    err_test = typical_err.test_score(4, 30, 5000, classifier)
    print("typical error = ", err_test)

    plotter(4, 30, tree)
