#%%
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  

import pandas as pd
import numpy as np

import string

#%%
def editDistance(word1, word2):
    """
    :type word1: str
    :type word2: str
    :rtype: int
    """
    n = len(word1)
    m = len(word2)
    
    # if one of the strings is empty
    if n * m == 0:
        return n + m
    
    # array to store the convertion history
    d = [ [0] * (m + 1) for _ in range(n + 1)]
    
    # init boundaries
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
       
    # DP compute 
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            left = d[i - 1][j] + 1
            down = d[i][j - 1] + 1
            left_down = d[i - 1][j - 1] 
            if word1[i - 1] != word2[j - 1]:
                left_down += 1
            d[i][j] = min(left, down, left_down)
        
    return d[n][m]


def calTfIdf(corpus):
    # print(len(corpus))
    vectorizer = CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频  
    transformer = TfidfTransformer()#该类会统计每个词语的tf-idf权值  
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵  
    word = vectorizer.get_feature_names()#获取词袋模型中的所有词语  
    weight = tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    return word, weight


def getWordList(sentenses):
    vocabulary = ""

    # eliminate punctuation
    sentenses = sentenses.replace(')', ' ')
    sentenses = sentenses.replace(']', ' ')
    sentenses = sentenses.replace('}', ' ')
    sentenses = ''.join(c for c in sentenses if c not in string.punctuation or c == '-')

    sentenses = sentenses.lower()
    vocabulary += sentenses
 
    word_list = vocabulary.split()
 
    return word_list



#%%
if __name__ == "__main__":
    print("load data") 
    # load data
    df = pd.read_csv("./ML3AllSites.csv", encoding="ISO-8859-1")
    # data = np.array(df.loc[:,:])
    # get column name
    column_headers = list(df.columns.values)

    '''
        deal with attentioncorrect
        Use GloVe to get word vector
        if contains instructions we think it is right and set to 1 else 0
    '''
    word_model = pd.read_csv("./word_vector_attention.csv", index_col=0, encoding="ISO-8859-1")

    template_word = word_model.loc['instructions', :]
    aaa = []
    for i in word_model.index:
        aaa.append([i, np.dot(template_word, word_model.loc[i, :])])
    def take_second(elem):
        return elem[1]
    aaa.sort(key=take_second, reverse=True)
    print(aaa[0:10])
    
    for i in range(len(df['attentioncorrect'])):
        paragraph = str(df['attentioncorrect'][i])
        
        if paragraph != 'nan':
            word_list = getWordList(paragraph)
            flag = True

            for j in range(len(word_list)):
                if np.dot(template_word, word_model.loc[word_list[j], :]) >= 3.2:
                    # contain word instructions
                    # use threshold to define the similiarity between words
                    df['attentioncorrect'][i] = 1
                    flag = False
                    break
            
            if flag:
                df['attentioncorrect'][i] = 0

    print("attentioncorrect complete") 
    
    '''
        drop useless column
    '''
    reserved_col = ['attentioncorrect', 'pate_01', 'pate_02']
    # tags need to drop
    to_drop = list(set(column_headers).difference(set(reserved_col)))
    df.drop(columns=to_drop, inplace=True)
    
    # store
    df.to_csv('ML3data_2.csv', header=reserved_col)
    # print(len(data[0][:]))

    print("mission complete")
