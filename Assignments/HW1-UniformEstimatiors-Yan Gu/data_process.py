import csv
import random

n = 100
L = 10
data = []
for j in range(1000):
    row = []
    for i in range(n):
        row.append(random.uniform(0, L))
    # print(row)
    data.append(row)
# print(np.array(data))

with open('data.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in data:
        spamwriter.writerow(row)