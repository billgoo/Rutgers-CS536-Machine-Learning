import pandas as pd
import math
from data_generator import DataGenerator
from navieRegression import LinearRegression


if __name__ == "__main__":
	filename = 'LinearRegression/data/question1_m_1000.csv'
	train_data = pd.read_csv(filename, names=['b','X1','X2','X3','X4','X5',
                                        	'X6','X7','X8','X9','X10',
                                        	'X11','X12','X13','X14','X15',
                                        	'X16','X17','X18','X19','X20','y'])

	# print(train_data)
	train_X, train_y = train_data[train_data.columns[:-1]].values, train_data[['y']].values
	print(train_X.shape, train_y.shape)
	# train
	model = LinearRegression()
	model.fit(train_X, train_y)
	print(model.param)
	train_err = 0.0
	for i in range(1000):
		train_err += math.pow((model.predict(train_X[i])-train_y[i]), 2)
	train_err /= 1000

	print('w:\n', model.param)
	print('train error:', train_err)

	#dg = DataGenerator(1)
	#dg.data_Generator(1000000)
	filename1 = 'LinearRegression/data/question1_m_1000000.csv'
	test_data = pd.read_csv(filename1, names=['b','X1','X2','X3','X4','X5',
                                        	'X6','X7','X8','X9','X10',
                                        	'X11','X12','X13','X14','X15',
                                        	'X16','X17','X18','X19','X20','y'])
	test_X, test_y = test_data[test_data.columns[:-1]].values, test_data[['y']].values
	test_err = 0.0
	for i in range(1000000):
		test_err += math.pow((model.predict(test_X[i])-test_y[i]), 2)
	test_err /= 1000000

	print('test error:', test_err)

