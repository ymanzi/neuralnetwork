import numpy as np


class MSE(object):
	"""
		def sigmoid_derivative(x):
			sig = sigmoid(x)
			return sig * (1.0 - sig)
	"""

	@staticmethod
	def value(a, y, weights, lambda_):
		"""Return the cost associated with an output ``a`` and desired output
		``y``.
		"""
		return 0.5* (np.linalg.norm(a-y)**2) / (a.shape[0] * a.shape[1])
		# return unregularize + (0.5*lambda_*np.sum(np.square(weights[-1])) / (a.shape[0] * a.shape[1])) 
		
	@staticmethod
	def delta(z, a, y):
		"""Return the error delta from the output layer."""
		return np.subtract(a, y) * sigmoid_derivative(z) ######## A MODIFIER

class CrossEntropyCost(object):

	@staticmethod
	def value(a, y, weights, lambda_):
		"""Return the cost associated with an output ``a`` and desired output
		``y``.  Note that np.nan_to_num is used to ensure numerical
		stability for log close to 0 values.  In particular, if both ``a`` and ``y`` have a 1.0
		in the same slot, then the expression (1-y)*np.log(1-a)
		returns nan.  The np.nan_to_num ensures that that is converted
		to the correct value (0.0).
		"""
		return np.sum(np.nan_to_num(-y*np.log(a + 1e-15)-(1-y)*np.log(1-a + 1e-15)))  / (a.shape[0] * a.shape[1])
		# return unregularize + (0.5*lambda_*np.sum(np.square(weights[-1])) / (a.shape[0] * a.shape[1])) 

	@staticmethod
	def delta(z, a, y):
		"""Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

		C = −[ylna+(1−y)ln(1−a)]
        """
		return np.subtract(a, y)