# use GloVe algorithm to get word vector

import numpy as np
import random
from math import log
import math
from scipy import spatial
import pandas as pd
import string
 
GlobalCost = 0

# deal with raw data
def buildVocabulary(raw_text):
    corpus = []
    vocabulary = ""
    for line in raw_text:
        # eliminate punctuation
        line = line.replace(')', ' ')
        line = line.replace(']', ' ')
        line = line.replace('}', ' ')
        line = ''.join(c for c in line if c not in string.punctuation or c == '-')

        line = line.lower()
        corpus.append(line)
        vocabulary += line + '\n '
 
    vocab = sorted(list(set(vocabulary.split())))
    # if ('big-boy' in vocab):
    #     print(True)
    # if ('coveredi' in vocab):
    #     print(True)
 
    return vocab, corpus


# build the co-occurence matrix
def buildCoocMatrix(vocab, corpus, window_size):
    # list stores paragraph
    paras = []
    
    for paragraph in corpus:
        # 2-D for [paragraph[words in each paragraph]]
        paras.append(paragraph.split())
 
    cooc = []
    serial_number = 1
    word_count = []

    # for each term present in the vocabulary
    for term in vocab:
        vectors = []

        # find all the indices of the current term wihtin the ordered list of words
        for word_list in paras:
            # get words in each answer (one answer as one paragraph)
            # because we select one para (one person's answer) each time
            # the context will just in the paragraph and not related to others' answers
            indices = [i for i, x in enumerate(word_list) if x == term]
 
            vector = [0 for j in range(0, len(vocab))]
 
            # find all left and right words in the context
            for i in indices:
                left_context = word_list[max(0, i - window_size): i]
                right_context = word_list[i + 1: min(i + 1 + window_size, len(word_list))]
 
                increament = len(left_context)

                for word in left_context:
                    try:
                        # skip the word if it is same as the word we are checking
                        if vocab.index(word) != i:
                            vector[vocab.index(word)] += (1.0 / increament)
                    except:
                        pass
                    increament -= 1
 
                increament = 1
                for word in right_context:
                    try:
                        if vocab.index(word) != i:
                            vector[vocab.index(word)] += (1.0 / increament)
                    except:
                        pass
                    increament += 1
            # appending each paragraph to the matrix
            vectors.append(vector)

        # since we have many paras for the same term as it may occur many times in the corpus
        # sum each column and create a resulting total vector
        temp = []
        for i in range(len(vocab)):
            m = 0
            for j in range(len(vectors)):
                m += vectors[j][i]

            # append each column to the temp list
            temp.append(m)

        # append the resulting vector to the co-occurence matrix
        cooc.append(temp)
        word_count.append(str(serial_number) + "\t" + term)
        if serial_number % 100 == 0:
            print(f"\t{str(serial_number)}\t{term}")
        serial_number += 1

    # store vocabulary and corresponding count for each word
    fout = open("./word_serial.txt", "w")
    for row in word_count:
        fout.write(row + "\n")
    fout.close()
 
    # extract non-zero elements to form a dense matrix
    M = []
    for i in range(len(vocab)):
        for j in range(len(vocab)):
            if cooc[i][j] != 0:
                M.append([i, j, cooc[i][j]])

    return M


# train model use GloVe algorithm and AdaGrad
def train_GloVe(vocab, cooc_matrix, iterations, vector_size, learning_rate, x_max, alpha):
    global GlobalCost
 
    # initialize weights and biases
    W = [[random.uniform(-0.5, 0.5) for i in range(vector_size)] for j in range(2 * len(vocab))]
    biases = [random.uniform(-0.5, 0.5) for i in range(2 * len(vocab))]
 
    # training is done using adaptive gradient descent (AdaGrad)
    # to make this work we need to store the sum of squares of all previous gradients
    # initialize the squared gradient weights and biases to 1
    grad_squared_w = [[1 for i in range(vector_size)] for j in range(2 * len(vocab))]
    grad_squared_biases = [1 for i in range(2 * len(vocab))]            
 
    for i in range(iterations):
        for main_id, context_id, data in cooc_matrix:
            w1 = W[main_id]
            w2 = W[context_id + len(vocab)]
            x = data
 
            # weighted function
            if x < x_max:
                f = (x / x_max) ** alpha
            else:
                f = 1
 
            inner_cost = (
                    np.dot(np.array(w1), np.array(w2)) + biases[main_id] + biases[context_id + len(vocab)] + log(data)
                )
            # calculate cost
            cost = f * (inner_cost ** 2)
 
            GlobalCost += 0.5 * cost
 
            # calculate the gradient for the word as both main and contextual
            grad_main = f * np.dot(inner_cost, np.array(w2))
            grad_context = f * np.dot(inner_cost, np.array(w1))
 
            # calculate the gradient of the bias for the word
            grad_bias_main = f * inner_cost
            grad_bias_context = f * inner_cost
 
            # applying AdaGrad
            for a in range(vector_size):
                w1[a] -= ((grad_main[a] * learning_rate) / math.sqrt(sum(grad_squared_w[main_id])))
                grad_squared_w[main_id][a] += grad_main[a] ** 2
 
            for a in range(vector_size):
                w2[a] -= ((grad_context[a] * learning_rate) / math.sqrt(sum(grad_squared_w[context_id + len(vocab)])))
                grad_squared_w[context_id + len(vocab)][a] += grad_context[a] ** 2
 
            biases[main_id] -= ((learning_rate * grad_bias_main) / math.sqrt(grad_squared_biases[main_id]))
            biases[context_id + len(vocab)] -= (
                    (learning_rate * grad_bias_context) / math.sqrt(grad_squared_biases[context_id + len(vocab)])
                )
 
            grad_squared_biases[main_id] += grad_bias_main ** 2
            grad_squared_biases[context_id + len(vocab)] += grad_bias_context ** 2

            W[main_id] = w1
            W[context_id + len(vocab)] = w2
 
        print(f"Iteration = {str(i)}, Cost = {str(GlobalCost)}.")
 
    return W


#%%
if __name__ == "__main__":
    '''
        global parameters
        window_size = 10 is each left and right context are 10 words and 20 in total,
        iterations = 100 is maximum training epoches using AdaGrad,
        vector_size = 10 is total number of weights 10 for every words,
        learning_rate, x_max, alpha are default
    '''
    window_size, iterations, vector_size, learning_rate, x_max, alpha = 10, 100, 10, 0.05, 100, 0.75

    # load data
    df = pd.read_csv("./ML3AllSites.csv", encoding = "ISO-8859-1")
    # get column name
    column_headers = list(df.columns.values)

    # form total corpus
    corpus = []
    index_age = []
    for i in range(len(df['highpower'])):
        if str(df['highpower'][i]) != 'nan':
            corpus.append(str(df['highpower'][i]))
    
    # generate vocabulary and processed corpus
    print("build vocabulary and corpus")
    vocab, corpus = buildVocabulary(corpus)

    # calculate co-occurence matrix
    print("build co-occurence matrix")
    cooc_matrix = buildCoocMatrix(vocab, corpus, window_size)

    # calculate word vector
    print("train model")
    W = train_GloVe(vocab, cooc_matrix, iterations, vector_size, learning_rate, x_max, alpha)
    Y = []
    
    # summation of the main and context vector for each word
    for i in range(len(vocab)):
        Y.append([(W[i][a] + W[i + len(vocab)][a]) for a in range(vector_size)])

    # store the word vector
    print("\n\nSave the word vector to csv.")
    Y = pd.DataFrame(Y, index=vocab)
    Y.to_csv('word_vector.csv')
    
    '''
        test and vector to word method
    '''
    '''
    #Find the 30 most similar words to the given word
    word = "teammate"
    index = vocab.index(word)
    print(vocab[index]+" : ")
    aaaa = np.array([0.8568,0.1290,0.5373,-0.4228,-0.8315,0.7872,-0.0654,-0.1867,0.1229,0.3215])
    #dists = np.dot(np.array(Y), np.array(Y[index]))
    # 向量相似度，方向越相同的点乘得到的结果越大
    dists = np.dot(np.array(Y), aaaa)
    z = list()
    for i in range(len(vocab)):
        z.append([dists[i], i])
    z = sorted(z, key=lambda x: x[0], reverse=True)
    for i in range(30):
        print(vocab[z[i][1]])
    '''
