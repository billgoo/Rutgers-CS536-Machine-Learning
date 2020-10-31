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
        if i % 5 == 0:
            plt.annotate("%.2f" % y_data[i], 
                        xy=(x_data[i],y_data[i]), xytext=(-20, 30), textcoords='offset points', color='red')

    plt.legend(loc='upper right')

    filename = 'images/question3/Figure.' + title[4] + '.png'

    # save the picture，filename is title
    plt.savefig(filename, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    m = 100
    k = 20
    eps = [i/100 for i in range(0, 101, 2)]
    iterations = 500
    counter = []
    dg = DataGenerator(3, iterations)
    for e in eps:
        dg.data_Generator(m, k, e)
    
    for e in eps:
        count = 0
        name_key = ['b']
        name_key.extend(['x' + str(i) for i in range(1, k + 1)])
        name_key.append('y')
        filename = 'data/question3/data_k_' + str(k) + '_m_' + str(m) + '_eps_' + str(e) + '.csv'
        dataset = pd.read_csv(filename, names=name_key)
        for i in range(iterations):
            data = np.array(dataset[i*100 : i*100+100].reset_index(drop=True))
            #print(np.array(data)[0])

            model = Perceptron(m, k, data)
            result = model.train()
            count += result[1]
        counter.append([e, count/iterations])
        print(e, count)

    # output the data to be re-format
    with open('data/question3/results.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in counter:
            spamwriter.writerow(row)

    # re-format and draw the xy-coordinate figure
    datamap = pd.read_csv('data/question3/results.csv', names=['epsilon','Steps'])
    col_e = datamap['epsilon']
    col_s = datamap['Steps']
    show_Picture(col_e, col_s, "Steps", "epsilon", "Steps need to terminate", 
                "Fig 3: Steps need to terminate for different epsilon.")
    