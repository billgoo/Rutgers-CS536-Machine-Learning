import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron
from data_generator import DataGenerator


def show_Picture(x_data, y_data, y_data_name, x_label, y_label, title):
    plt.figure(figsize=(16, 8))

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.plot(x_data, y_data, marker='.', c='red', lw=0.5, label=y_data_name)
    
    for i in range(len(x_data)):
        if i % 4 == 0:
            plt.annotate("%.2f" % y_data[i], 
                        xy=(x_data[i],y_data[i]), xytext=(-20, 30), textcoords='offset points', color='red')

    plt.legend(loc='upper right')

    filename = 'images/question4/Figure.' + title[4:8] + '.png'

    # save the picture，filename is title
    plt.savefig(filename, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    m = [100, 1000]
    k = list(range(2, 41))
    eps = 1
    iterations = 500
    counter_m = dict()
    
    dg = DataGenerator(4, iterations)
    for i in k:
        for j in m:
            dg.data_Generator(j, i, eps)
            
    
    for mi in m:
        counter = []
        for ki in k:
            count = 0
            name_key = ['b']
            name_key.extend(['x' + str(i) for i in range(1, ki + 1)])
            name_key.append('y')
            filename = 'data/question4/data_k_' + str(ki) + '_m_' + str(mi) + '_eps_' + str(eps) + '.csv'
            dataset = pd.read_csv(filename, names=name_key)
            for i in range(iterations):
                data = np.array(dataset[i*mi : i*mi+mi].reset_index(drop=True))
                #print(np.array(data)[0])

                model = Perceptron(mi, ki, data)
                result = model.train()
                count += result[1]
            counter.append([ki, count/iterations])
            print(ki, count)
        counter_m[mi] = counter

    # output the data to be re-format
    with open('data/question4/results_100.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in counter_m[m[0]]:
            spamwriter.writerow(row)

    # output the data to be re-format
    with open('data/question4/results_1000.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in counter_m[m[1]]:
            spamwriter.writerow(row)

    # re-format and draw the xy-coordinate figure
    for mi in m:
        fname = 'data/question4/results_' + str(mi) + '.csv'
        datamap = pd.read_csv(fname, names=['k','Steps'])
        col_k = datamap['k']
        col_s = datamap['Steps']
        figname = "Fig 4(" + str(m.index(mi)) + "): Steps need to terminate for different k with m = " + str(mi)
        show_Picture(col_k, col_s, "Steps", "k", "Steps need to terminate", figname)
    