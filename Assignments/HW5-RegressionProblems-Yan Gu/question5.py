import csv
import random
import numpy as np
import math
from collections import Counter

def data_Generator(m, w, b, var, iteration):
    data = []
    if m == 0:
        return data

    for i in range(m * iteration):
        x = random.uniform(100, 102)
        eps = np.random.normal(0, math.sqrt(var))
        y = w * x + b + eps
        x_sift = x - 101

        xy = [x, x_sift, y, eps]            

        data.append(xy)

    filename = 'data/data.csv'
    with open(filename, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in data:
            spamwriter.writerow(row)
       
    return data



if __name__ == "__main__":
    m = 200
    w = 1
    b = 5
    var = 0.1
    iteration = 1000
    data = data_Generator(m, w, b, var, iteration)

    w_and_b = []

    for i in range(iteration):
        sub_data = data[i*200:i*200+200]
        Sigma = np.sum(sub_data, axis=0)
        Sig_x, Sig_x_sift,  Sig_y = Sigma[0], Sigma[1], Sigma[2]

        Sig_x_square = 0.0
        Sig_x_square_sift = 0.0
        Sig_xy = 0.0
        Sig_xy_sift = 0.0
        for j in range(m):
            Sig_x_square += pow(sub_data[j][0], 2)
            Sig_x_square_sift += pow(sub_data[j][1], 2)
            Sig_xy += sub_data[j][0] * sub_data[j][2]
            Sig_xy_sift += sub_data[j][1] * sub_data[j][2]

        det = m * Sig_x_square - pow(Sig_x, 2)
        det_sift = m * Sig_x_square_sift - pow(Sig_x_sift, 2)

        a, d = Sig_x_square / det, m / det
        b = c = -Sig_x / det
        a_sift, d_sift = Sig_x_square_sift / det_sift, m / det_sift
        b_sift = c_sift = -Sig_x_sift / det_sift
    
        w_bar = a * Sig_y + b * Sig_xy
        b_bar = c * Sig_y + d * Sig_xy
        w_bar_sift = a_sift * Sig_y + b_sift * Sig_xy_sift
        b_bar_sift = c_sift * Sig_y + d_sift * Sig_xy_sift
        w_and_b.append([w_bar, b_bar, w_bar_sift, b_bar_sift])

    filename = 'data/result.csv'
    with open(filename, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in data:
            # each row is: w, b, w', b'
            spamwriter.writerow(row)

    expect, variance = np.mean(w_and_b, axis=0), pow(np.std(w_and_b, axis=0), 2)
    print(" \tw\tb\tw_sift\tb_sift")
    print("expect", expect[0], expect[1], expect[2], expect[3])
    print("variance", variance[0], variance[1], variance[2], variance[3])
