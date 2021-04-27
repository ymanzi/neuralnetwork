import numpy as np

class Weight_init:
	@staticmethod
	def std(layers):
		return [np.random.randn(x, y) for x,y in zip(layers[1:], layers[:-1])]

	@staticmethod
	def xavier(layers):
		return [np.random.randn(x, y) / np.sqrt(y + x) for x,y in zip(layers[1:], layers[:-1])]

	@staticmethod
	def he(layers):
		return [np.random.randn(x, y) / np.sqrt((y + x) / 2) for x,y in zip(layers[1:], layers[:-1])]