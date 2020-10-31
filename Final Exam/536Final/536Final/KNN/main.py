import csv
import pandas as pd
import numpy as np
from knn import knn_impute

# # splitting the dataset into 80% training - 20% validation
# def split_data(data):
#     percentage = 0.8
#     split = int(len(data)*percentage)
#     train_data = data.iloc[:split, :]
#     val_data = data.iloc[split:, :]
#     return train_data, val_data


def exp_1():
    data = pd.read_csv('data/ML3data.csv')
    #data.drop(['Unnamed: 0'], axis=1, inplace=True)
    useful_set = data[['mcmost1', 'mcmost2', 'mcmost3', 'mcmost4', 'mcmost5', \
                        'mcsome1', 'mcsome2', 'mcsome3', 'mcsome4', 'mcsome5', \
                        'mcdv1', 'mcdv2']]
    
    # delete rows with no feature at all
    useful_set = useful_set.dropna(how='all')
    target = useful_set[['mcdv1']]
    val = useful_set[['mcdv1']]
    #print(target)
    original_att = useful_set[['mcmost1', 'mcmost2', 'mcmost3', 'mcmost4', 'mcmost5', \
                             'mcsome1', 'mcsome2', 'mcsome3', 'mcsome4', 'mcsome5', \
                             'mcdv2']]
    attributes = useful_set[['mcmost1', 'mcmost2', 'mcmost3', 'mcmost4', 'mcmost5', \
                             'mcsome1', 'mcsome2', 'mcsome3', 'mcsome4', 'mcsome5', \
                             'mcdv2']]
    knn_impute(target, attributes, 9, aggregation_method="mode")

  
    null_matrix = val.isnull()
    
    #print(val)
    for i, row in val.iterrows():
        if pd.isnull(val.loc[i]['mcdv1']):
            print("attributes:\n",original_att.loc[i])
            print("prediction:", target.loc[i]['mcdv1'])
            print('==================')
            
        
def exp_2():
    data = pd.read_csv('data/ML3data_2.csv')
    useful_set = data[['attentioncorrect', 'pate_01', 'pate_02']]
    useful_set = useful_set.dropna(how='all')

    target = useful_set[['pate_01']]
    val = useful_set[['pate_01']]

    attributes = useful_set[['pate_02', 'attentioncorrect']]
    original_att = useful_set[['pate_02', 'attentioncorrect']]
    knn_impute(target, attributes, 5, aggregation_method="mode")

    for i, row in val.iterrows():
        if pd.isnull(val.loc[i]['pate_01']):
            print("attributes:\n",original_att.loc[i])
            print("prediction:", target.loc[i]['pate_01'])
            print('==================')



    

if __name__ == '__main__':
    #exp_1()
    exp_2()




