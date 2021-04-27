import numpy as np

def mean_(x):
	length = x.size
	if length == 0:
		return None
	return float(np.sum(x) / length)

def median_(x):
	length = x.size
	if length == 0:
		return None
	y = np.sort(x)
	if (length % 2):
		ret = y[int(length / 2)]
	else:
		ret = (y[int(length / 2) - 1] + y[int(length / 2)]) / 2
	return float(ret)

def quartiles_(x, percentile):
	length = x.size
	if length == 0:
		return None
	y = np.sort(x)
	per = (percentile / 100) * length
	if (per % 1.0):
		return float(y[int(per)])
	else:
		return float(y[int(per) - 1])

def var_(x):
	length = x.size
	if length == 0:
		return None
	mean = float(np.sum(x) / length)
	return np.sum(np.power(np.subtract(x, mean), 2)) / length

def std_(x):
	length = x.size
	if length == 0:
		return None
	mean = float(np.sum(x) / length)
	return np.sqrt(np.sum(np.power(np.subtract(x, mean), 2)) / length)
