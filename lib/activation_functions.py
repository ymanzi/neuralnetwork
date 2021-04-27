import numpy as np

class Softmax:
	@staticmethod
	def fct(x: np.ndarray) -> np.ndarray:
		vec = np.exp(x)
		return vec / np.sum(vec)
	
	@staticmethod
	def derivative(x):
		sm = Softmax.fct(x)
		return sm * (1.0 - sm)

class Sigmoid:
	@staticmethod
	def fct(x: np.ndarray) -> np.ndarray:
		if x.size == 0:
			return None
		x = x.astype(float)
		if x.ndim == 0:
			x = np.array(x, ndmin=1)
		return (1.0 / (1.0 + (np.exp(-x))))

	@staticmethod
	def derivative(x):
		sig = Sigmoid.fct(x)
		return sig * (1.0 - sig)

class Tanh:
	@staticmethod
	def fct(x: np.ndarray) -> np.ndarray:
		pos_val = np.exp(x)
		neg_val = np.exp(-x)
		return (pos_val - neg_val) / (pos_val + neg_val)

	@staticmethod
	def derivative(x):
		t = Tanh.fct(x)
		return 1.0 - t**2

class ReLu:
	@staticmethod
	def fct(x: np.ndarray) -> np.ndarray:
		return np.where(x < 0, np.exp(x) - 1, x)
		# return np.maximum(- 0.01 * x, x)
	
	@staticmethod
	def derivative(x):
		return np.where(x < 0, np.exp(x), 1)