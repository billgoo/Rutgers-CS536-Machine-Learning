import pandas as pd
import numpy as np
from pprint import pprint
import random

class DecisionTreeClassifier:

    def __init__(self, k, m, dataset):
        self.k = k
        self.m = m
        self.dataset = dataset
        self.tree = {}
        self.tree_with_data = {}
    

    def entropy(self, target_col):
        """
        Calculate the entropy of a dataset.
        The only parameter of this function is the target_col parameter which specifies the target column
        """
        elements, counts = np.unique(target_col, return_counts = True)
        entropy = np.sum([(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
        return entropy
     
        
    
    def info_Gain(self, data, split_attribute_name, target_name = "Y"):
        """
        It is a function that takes a data set and a variable, and returns the infomation gain of the variable 
        which will then use to partition the data set..
        Calculate the information gain of a dataset. This function takes three parameters:
        1. data = The dataset for whose feature the IG should be calculated
        2. split_attribute_name = the name of the feature for which the information gain should be calculated
        3. target_name = the name of the target feature. The default for this example is "class"
        """    
        # Calculate the entropy of the total dataset
        total_entropy = self.entropy(data[target_name])
        
        # Calculate the values and the corresponding counts for the split attribute 
        values, counts= np.unique(data[split_attribute_name], return_counts = True)
        
        # Calculate the weighted entropy
        weighted_entropy = np.sum([(counts[i] / np.sum(counts)) *
                                self.entropy(data.where(data[split_attribute_name] == values[i]).dropna()[target_name]) 
                                for i in range(len(values))])
        
        # Calculate the information gain
        info_gain = total_entropy - weighted_entropy
        return info_gain
        
    
    # build ID3 tree
    def fit_ID3(self, data_tree, data, originaldata, features, target_attribute_name = "Y", parent_node_class = None):
        """
        ID3 Algorithm: This function takes five paramters:
        1. data = the data for which the ID3 algorithm should be run. In the first run this equals the total dataset,
        then the sub set of data. We can just attach it with the node in the tree we will print, so the tree is with the data.
        2. originaldata = This is the original dataset needed to calculate the mode target feature value of the original dataset
        in the case the dataset delivered by the first parameter is empty. It is the total dataset.
        3. features = the feature space of the dataset . This is needed for the recursive call since during the tree growing process
        we have to remove features from our dataset. Splitting at each node.
        4. target_attribute_name = the name of the target attribute, in our problem it is Y so I set it in advance without passing the value.
        5. parent_node_class = This is the value or class of the mode target feature value of the parent node for a specific node. This is 
        also needed for the recursive call since if the splitting leads to a situation that there are no more features left in the feature
        space, we want to return the mode target feature value of the direct parent node.
        """   
        # Define the stopping criteria: If one of this is satisfied, we want to return a leaf node.

        if len(np.unique(data[target_attribute_name])) <= 1:
            # If all target_values have the same value, return this value
            return np.unique(data[target_attribute_name])[0]
        elif len(data) == 0:
            #If the sub dataset is empty, return the mode target feature value in the original dataset
            return np.unique(originaldata[target_attribute_name])[
                np.argmax(np.unique(originaldata[target_attribute_name], return_counts = True)[1])]
        elif len(features) == 0:
            # If the feature space is empty, return the mode target feature value of the direct parent node --> Note that
            # the direct parent node is that node which has called the current run of the ID3 algorithm and hence
            # the mode target feature value is stored in the parent_node_class variable.
            return parent_node_class
        else:
            # If none of the above holds true, grow the tree.

            # Set the default value for this node --> The mode target feature value of the current node
            parent_node_class = np.unique(data[target_attribute_name])[
                np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
            
            # Select the feature which best splits the dataset - max information gain
            # Return the information gain values for the features in the dataset
            item_values = [self.info_Gain(data, feature, target_attribute_name) for feature in features]
            best_feature_index = np.argmax(item_values)
            best_feature = features[best_feature_index]
            
            # Create the tree structure. The root gets the name of the feature (best_feature)
            # with the maximum information gain in the first run
            tree = {best_feature:{}}
            # Create the tree with data. The root gets the name of the feature (best_feature)
            # with the maximum information gain in the first run and the data and child in dict as two key
            data_tree[best_feature] = {"data":data, "child":{}}
            
            # Remove the feature with the best inforamtion gain from the feature space
            features = [i for i in features if i != best_feature]
            
            # Grow a branch under the root node for each possible value of the root node feature
            for value in np.unique(data[best_feature]):
                value = value
                # Split the dataset along the value of the feature with the largest information gain and create sub_datasets
                sub_data = data.where(data[best_feature] == value).dropna()

                # Initial the tree with data structure with each key of its child
                # then change the child in the recursion
                data_tree[best_feature]["child"][value] = {}
                
                # Recursive call the ID3 algorithm for each of those sub_datasets with the new parameters
                subtree = self.fit_ID3(data_tree[best_feature]["child"][value], sub_data, self.dataset, features,
                                        target_attribute_name, parent_node_class)
                
                # Add the sub tree, grown from the sub_dataset to the tree under the root node
                tree[best_feature][value] = subtree
                
            self.tree = tree
            self.tree_with_data = data_tree
            return tree
    
        
    # predict just using tree for both algorithms
    def predict(self, data_tree, query, tree):
        '''
        Use the tree to predict
        If we not find a leaf for the input data, then we will find the most possible value
        given the existing variables.
        '''

        for key in list(query.keys()):
            if key in list(tree.keys()):
                # if not find a leaf
                try:
                    result = tree[key][query[key]]
                    # tree_data is the tree with data
                    # the root is the exact variable in the last recursion
                except:
                    # from the tree we get the one with larger P(Y = y)
                    data_tree = data_tree[key]["data"]["Y"]
                    count = data_tree.value_counts()
                    if count[0] > count[1]:
                        return 0
                    elif count[0] < count[1]:
                        return 1
                    # else randomly choose whether is 0 or 1
                    return random.choice([0, 1])
    
                # get sub tree at variable = key, variable's value = query[key]
                result = tree[key][query[key]]
                
                # recursion
                if isinstance(result, dict):
                    return self.predict(data_tree[key]["child"][query[key]], query, result)
                else:
                    return result
            

    # return predict accuracy for both algorithms
    def score(self, data):
        # Create new query instances by simply removing the target feature column from the original dataset and 
        # convert it to a dictionary
        features = data.iloc[:,:-1].to_dict(orient = "records")
        
        # Create a empty DataFrame in whose columns the prediction of the tree are stored
        predicted = pd.DataFrame(columns=["predict"]) 
        
        # Calculate the prediction accuracy
        for i in range(len(data)):
            '''
            if i > 0 and i % 10000 == 0:
                print("10000 iters")
            '''
            predicted.loc[i, "predict"] = self.predict(self.tree_with_data, features[i], self.tree) 
        # print('The prediction error is: ',(np.sum(predict["predict"] != data["Y"])/len(data))*100,'%')
        return (np.sum(predicted["predict"] != data["Y"])/len(data))
    
    
    #计算数据集的基尼指数
    def gini_D(self, data, target_name = "Y"):
        total = len(data[target_name])
        values, counts= np.unique(data[target_name], return_counts = True)
        imp = 0.0
        for key1 in range(len(values)):
            prob1 = float(counts[key1]) / total  
            for key2 in range(len(values)):
                if key1 == key2: continue
                prob2 = float(counts[key2]) / total
                imp += prob1 * prob2
        return imp


    #计算数据集的基尼指数
    def gini(self, data, split_attribute_name, target_name = "Y"):
        total = len(data[target_name])
        elements, counts = np.unique(data[split_attribute_name], return_counts = True)
        #counts = uniqueCounts(data[split_attribute_name])
        # gini for feature R
        gini_D_R = {}
        n = len(elements)

        for i in range(n):
            sub_data = data.where(data[split_attribute_name] == elements[i]).dropna()
            gini_D_R[i] = self.gini_D(sub_data)
            
        imp = 0.0

        for key in range(n):
            prob = float(counts[key]) / total
            imp += prob * gini_D_R[i]

        return imp


    # build CART tree
    def fit_CART(self, data_tree, data, originaldata, features, target_attribute_name = "Y", parent_node_class = None):
        """
        ID3 Algorithm: This function takes five paramters:
        1. data = the data for which the ID3 algorithm should be run. In the first run this equals the total dataset,
        then the sub set of data. We can just attach it with the node in the tree we will print, so the tree is with the data.
        2. originaldata = This is the original dataset needed to calculate the mode target feature value of the original dataset
        in the case the dataset delivered by the first parameter is empty. It is the total dataset.
        3. features = the feature space of the dataset . This is needed for the recursive call since during the tree growing process
        we have to remove features from our dataset. Splitting at each node.
        4. target_attribute_name = the name of the target attribute, in our problem it is Y so I set it in advance without passing the value.
        5. parent_node_class = This is the value or class of the mode target feature value of the parent node for a specific node. This is 
        also needed for the recursive call since if the splitting leads to a situation that there are no more features left in the feature
        space, we want to return the mode target feature value of the direct parent node.
        """   
        # Define the stopping criteria: If one of this is satisfied, we want to return a leaf node.

        if len(np.unique(data[target_attribute_name])) <= 1:
            # If all target_values have the same value, return this value
            return np.unique(data[target_attribute_name])[0]
        elif len(data) == 0:
            #If the sub dataset is empty, return the mode target feature value in the original dataset
            return np.unique(originaldata[target_attribute_name])[
                np.argmax(np.unique(originaldata[target_attribute_name], return_counts = True)[1])]
        elif len(features) == 0:
            # If the feature space is empty, return the mode target feature value of the direct parent node --> Note that
            # the direct parent node is that node which has called the current run of the ID3 algorithm and hence
            # the mode target feature value is stored in the parent_node_class variable.
            return parent_node_class
        else:
            # If none of the above holds true, grow the tree.

            # Set the default value for this node --> The mode target feature value of the current node
            parent_node_class = np.unique(data[target_attribute_name])[
                np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
            
            # Select the feature which best splits the dataset - max information gain
            # Return the information gain values for the features in the dataset
            item_values = [self.gini(data, feature, target_attribute_name) for feature in features]
            best_feature_index = np.argmin(item_values)
            best_feature = features[best_feature_index]
            
            # Create the tree structure. The root gets the name of the feature (best_feature)
            # with the maximum information gain in the first run
            tree = {best_feature:{}}
            # Create the tree with data. The root gets the name of the feature (best_feature)
            # with the maximum information gain in the first run and the data and child in dict as two key
            data_tree[best_feature] = {"data":data, "child":{}}
            
            # Remove the feature with the best inforamtion gain from the feature space
            features = [i for i in features if i != best_feature]
            
            # Grow a branch under the root node for each possible value of the root node feature
            for value in np.unique(data[best_feature]):
                value = value
                # Split the dataset along the value of the feature with the largest information gain and create sub_datasets
                sub_data = data.where(data[best_feature] == value).dropna()

                # Initial the tree with data structure with each key of its child
                # then change the child in the recursion
                data_tree[best_feature]["child"][value] = {}
                
                # Recursive call the ID3 algorithm for each of those sub_datasets with the new parameters
                subtree = self.fit_ID3(data_tree[best_feature]["child"][value], sub_data, self.dataset, features,
                                        target_attribute_name, parent_node_class)
                
                # Add the sub tree, grown from the sub_dataset to the tree under the root node
                tree[best_feature][value] = subtree
                
            self.tree = tree
            self.tree_with_data = data_tree
            return tree



if __name__ == "__main__":
    pass