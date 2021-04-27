import numpy as np
import pandas as pd
from lib.utils import *

def split_data(filename):
	df = pd.read_csv(filename, header=None).dropna().drop_duplicates()
	y = df[1].replace(['B', 'M'], [0, 1])
	x = df.drop(columns = [1, 4, 5]) #2 4
	data = data_spliter(x.values, y.values.reshape(-1, 1), 0.5) # , np.array(y_train).reshape(-1, 1), 0.6)
	pd.DataFrame(data[0]).to_csv("resources/x_train.csv", header = None, index=False)
	pd.DataFrame(data[1]).to_csv("resources/y_train.csv", header = None, index=False)
	pd.DataFrame(data[2]).to_csv("resources/x_test.csv", header = None, index=False)
	pd.DataFrame(data[3]).to_csv("resources/y_test.csv", header=None, index=False)

def init_project_train(filename):
	''' Initialize the lists data_train and data_test for the project from a given file'''
	
	df = pd.read_csv(filename, header=None).dropna().drop_duplicates()
	y = df[1].replace(['B', 'M'], [0, 1])
	x = df.drop(columns = [1, 4, 5]) #4 5
	data = data_spliter(x.values, y.values.reshape(-1, 1), 0.7) # , np.array(y_train).reshape(-1, 1), 0.6)
	
	x_train = init_array(data[0], 'x')
	y_train = init_array(data[1], 'y')
	x_test = init_array(data[2], 'x')
	y_test = init_array(data[3], 'y')
	
	data_train = list(zip(x_train, y_train))
	data_test = list(zip(x_test, y_test))
	return data_train, data_test


def init_project_predict(filename):
	''' Initialize the lists data_train and data_test for the project from a given file'''
	
	df = pd.read_csv(filename, header=None).dropna().drop_duplicates()
	y = df[1].replace(['B', 'M'], [0, 1]).values.reshape(-1, 1)
	x = df.drop(columns = [1, 4, 5]).values #2 4
	
	x_train = init_array(x, 'x')
	y_train = init_array(y, 'y')
	
	data_train = list(zip(x_train, y_train))
	return data_train


def split_validation(file_x, file_y):
	x = pd.read_csv(file_x, header=None)
	y = pd.read_csv(file_y, header=None)

	data = data_spliter(x.values, y.values.reshape(-1, 1), 0.5) # , np.array(y_train).reshape(-1, 1), 0.6)
	pd.DataFrame(data[0]).to_csv("resources/x_test.csv", header = None, index=False)
	pd.DataFrame(data[1]).to_csv("resources/y_test.csv", header = None, index=False)
	pd.DataFrame(data[2]).to_csv("resources/x_validation.csv", header = None, index=False)
	pd.DataFrame(data[3]).to_csv("resources/y_validation.csv", header=None, index=False)

def init_array(arr , kind):
	" init the data-shape to make it usable for the NN"

	# arr = pd.read_csv(file, header=None).dropna().values
	arr = array_normalization(arr)
	if kind == 'x':
		arr = x_array_reshape(arr)
	elif kind == 'y':
		arr = y_array_reshape(arr)
	return arr

def array_normalization(arr):
	for i in range(arr.shape[1]):
		# arr[:,i] = minmax_normalization(arr[:,i])
		arr[:,i] = zscore_normalization(arr[:,i])
	return arr

def x_array_reshape(arr):
	return np.array([elem.reshape(-1, 1) for elem in arr])

def y_array_reshape(arr):
	arr = arr.reshape(-1, 1)
	tmp_arr = [[0, 1] if elem == 1 else [1, 0] for elem in arr]
	return np.array([np.array(elem).reshape(-1, 1) for elem in tmp_arr])



# split_data("data.csv")