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
        deal with age
        get the most possible number in the answers
        if not NaN but there is no number then set to 0 (may be not so correct but useful)
    '''
    corpus_age = []
    index_age = []
    for i in range(len(df['age'])):
        if str(df['age'][i]) != 'nan':
            corpus_age.append(str(df['age'][i]))
            index_age.append(i)
    # print(len(corpus))
    # total word dictionary and weights
    word_age, weight_age = calTfIdf(corpus_age)

    for i in range(len(weight_age)):
        # modify index_age[i] row of the age column
        # the keyword with max probability
        max_key = ''
        max_p = 0.0
        for j in range(len(word_age)):
            if max_p < weight_age[i][j]:
                try:
                    # whether the keyword is a number
                    max_key = int(word_age[j])
                except:
                    # not number then try next probable keyword
                    continue
                max_key = word_age[j]
                max_p = weight_age[i][j]
        # whether we finally get the valid age, if not set to 0
        try:
            max_key = int(max_key)
        except:
            max_key = 0
        df['age'][i] = max_key
    print("age complete") 

    '''
    deal with anagram
    '''
    anagrams = {'anagrams1': 'trypa', 'anagrams2': 'aaflt', 'anagrams3': 'oneci', 'anagrams4': 'acelo'}

    for key in anagrams:
        col_len = len(df)
        for i in range(col_len):
            if str(df[key][i]) != 'nan':
                df[key][i] = editDistance(df[key][i], anagrams[key])
                # print(df[key][i])
    print("anagram complete") 

    '''
        deal with backcount
        just true or false, if not true calculation, others assigned to false
        True = 1, False = 0
    '''
    origin_num = 360
    origin_tag = 'backcount'
    for idx in range(1, 11):
        cur_tag = origin_tag + str(idx)
        cur_num = origin_num - 3 * idx
        for i in range(len(df[cur_tag])):        
            if str(df[cur_tag][i]) != 'nan':
                try:
                    if int(df[cur_tag][i]) == cur_num:
                        df[cur_tag][i] = 1
                    else:
                        df[cur_tag][i] = 0
                except:
                    df[cur_tag][i] = 0
    print("backcount complete")

    '''
        deal with gender
        just use numbers to represent
        1: Female, 2: Male, 3: not to answer, 4: Agender,
        5: Androgyne, 6: Bigender, 7: Cis, 8: Female to Male,
        9: Gender Fluid, 10: Gender Nonconforming,
        11: Gender Questioning, 12: Pangender, 13: Trans,
        14: Transsexual, 15: Two-spirit, 16: Other
    '''
    gender_dict = {4: {'agender'}, 5: {'androgyne', 'androgynous'}, 6: 'bigender', 
                    7: {'cis', 'cis female', 'cis woman', 'cis male', 'cis man', 'cisgender',
                        'cisgender female', 'cisgender male', 'cisgender man', 'cisgender woman'},
                    8: {'female to male', 'ftm'}, 9: {'gender fluid'}, 10: {'gender nonconforming'},
                    11: {'gender questioning', 'gender variant', 'genderqueer', 'intersex', 
                        'male to female', 'mtf', 'neither', 'neutrois', 'non-binary'}, 
                    12: {'pangender'}, 13: {'trans', 'trans female', 'trans male', 'trans man',
                        'trans person', 'trans woman', 'trans*', 'trans* female', 'trans* male',
                        'trans* man', 'trans* person', 'trans* woman', 'transfeminine', 'transgender',
                        'transgender female', 'transgender male', 'transgender man', 'transgender person',
                        'transgender woman', 'transmasculine'}, 
                    14: {'transsexual', 'transsexual female', 'transsexual male', 'transsexual man',
                        'transsexual person', 'transsexual woman'}, 15: {'two-spirit'}, 16: {'other'}}

    for i in range(len(df['gender'])):
        if str(df['gender'][i]) == 'nan':
            continue
        elif str(df['gender'][i]) == '1' or str(df['gender'][i]) == '2' or str(df['gender'][i]) == '3':
            df['gender'][i] = int(df['gender'][i])
            continue
        else:
            for k in gender_dict:
                if str(df['gender'][i]).lower() in gender_dict[k]:
                    df['gender'][i] = k
                    break
                if k == 16:
                    df['gender'][i] = k
    print("gender complete")

    '''
        deal with highpower
        Use GloVe to get word vector
        averaging the word vector in each answer to get paragraph vector
    '''
    word_model = pd.read_csv("./word_vector.csv", index_col=0, encoding="ISO-8859-1")
    
    for i in range(len(df['highpower'])):
        paragraph = str(df['highpower'][i])
        
        if paragraph != 'nan':
            word_list = getWordList(paragraph)

            # averaging to transfer paragragh to vector
            para_vector = np.array([0 for ii in range(10)], dtype=np.float64)
            for j in range(len(word_list)):
                para_vector += np.array(word_model.loc[word_list[j], :])

            para_vector /= len(word_list)

            df['highpower'][i] = para_vector
    print("highpower complete") 
    
    '''
        drop useless column
    '''
    reserved_col = ['age', 'anagrams1', 'anagrams2', 'anagrams3', 'anagrams4', 'backcount1', 'backcount2',
                    'backcount3', 'backcount4', 'backcount5', 'backcount6', 'backcount7', 'backcount8',
                    'backcount9', 'backcount10', 'big5_01', 'big5_02', 'big5_03', 'big5_04', 'big5_05',
                    'big5_06', 'big5_07', 'big5_08', 'big5_09', 'big5_10', 'elm_01', 'elm_02', 'elm_03',
                    'elm_04', 'elm_05', 'gender', 'highpower', 'intrinsic_01', 'intrinsic_02', 'intrinsic_03',
                    'intrinsic_04', 'intrinsic_05', 'intrinsic_06', 'intrinsic_07', 'intrinsic_08',
                    'intrinsic_09', 'intrinsic_10', 'intrinsic_11', 'intrinsic_12', 'intrinsic_13',
                    'intrinsic_14', 'intrinsic_15', 'mood_01', 'mood_02', 'nfc_01', 'nfc_02', 'nfc_03',
                    'nfc_04', 'nfc_05', 'nfc_06', 'pate_01', 'pate_02', 'pate_03', 'pate_04', 'pate_05',
                    'selfesteem_01', 'stress_01', 'stress_02', 'stress_03', 'stress_04']
    # tags need to drop
    to_drop = list(set(column_headers).difference(set(reserved_col)))
    df.drop(columns=to_drop, inplace=True)
    
    # store
    df.to_csv('ML3data.csv', header=reserved_col)
    # print(len(data[0][:]))

    print("mission complete")
