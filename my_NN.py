import numpy as np
import random

"IMPLEMENT VARIABLE LEARNING RATE"

def sigmoid(x: np.ndarray) -> np.ndarray:
	if x.size == 0:
		return None
	x = x.astype(float)
	if x.ndim == 0:
		x = np.array(x, ndmin=1)
	return (1.0 / (1.0 + (np.exp(-x))))

def sigmoid_derivative(x):
	sig = sigmoid(x)
	return sig * (1.0 - sig)

class QuadraticCost(object):
	"""
		def sigmoid_derivative(x):
			sig = sigmoid(x)
			return sig * (1.0 - sig)
	"""

	@staticmethod
	def value(a, y):
		"""Return the cost associated with an output ``a`` and desired output
		``y``.
		"""
		return 0.5*np.linalg.norm(a-y)**2
		
	@staticmethod
	def delta(z, a, y):
		"""Return the error delta from the output layer."""
		return np.subtract(a, y) * sigmoid_derivative(z)

class CrossEntropyCost(object):

	@staticmethod
	def value(a, y):
		"""Return the cost associated with an output ``a`` and desired output
		``y``.  Note that np.nan_to_num is used to ensure numerical
		stability for log close to 0 values.  In particular, if both ``a`` and ``y`` have a 1.0
		in the same slot, then the expression (1-y)*np.log(1-a)
		returns nan.  The np.nan_to_num ensures that that is converted
		to the correct value (0.0).

		"""
		return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

	@staticmethod
	def delta(z, a, y):
		"""Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

		C = −[ylna+(1−y)ln(1−a)]
        """
		return np.subtract(a, y)

class Network(object):
	def __init__(self, layers, cost=QuadraticCost):
		''' 
			Exemple of size: [2, 3, 1] 
			if we want to create a Network object with 
				2 neurons in the first layer, 
				3 neurons in the second layer, and 
				1 neuron in the final layer

			in the weights initialization we divide by 'np.sqrt(x)'
			to minimize the value of z because if the activation function
			is sigmoid and we don't do that and x is a large number of vector,
			there are a lot of chances that z will be large numbers and sigmoid(z)
			will saturate in the beginning.
			As the sigmoid graph show it, if z << 0 or z >> 1, a small change in the input
			give a small change in the output, we say the neuron is 'saturated'
		'''
		self.layers = layers
		self.nb_layers = len(layers)
		self.weights = [np.random.randn(x, y)/np.sqrt(x) for x,y in zip(layers[1:], layers[:-1])]
		self.biases = [np.random.randn(x, 1) for x in layers[1:]]
		self.cost = cost

	def quadratic_cost_derivative(self, output_activations, y):
		return np.subtract(output_activations, y)

	def feedforward(self, a):
		"""Return the output of the network if "a" is input.
			a′=σ(wa+b)
		"""
		for weight, bias in zip(self.weights, self.biases):
			a = sigmoid(np.add(np.dot(weight, a), bias))
		return a

	def backpropagation(self, x, y):
		"""
		Return a tuple ``(nabla_b, nabla_w)`` representing the
		gradient for the cost function C_x.  ``nabla_b`` and
		``nabla_w`` are layer-by-layer lists of numpy arrays, similar
		to ``self.biases`` and ``self.weights``.

		z = wa + b
		a′=σ(wa+b)  activation function
		"""
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		nabla_b = [np.zeros(b.shape) for b in self.biases]

		#feedforward
		list_activation = [x]
		list_z = []
		a = x
		for weight, bias in zip(self.weights, self.biases):
			z = np.add(np.dot(weight, a), bias)
			a = sigmoid(z)
			list_activation.append(a)
			list_z.append(z)
		delta = self.cost.delta(list_z[-1], list_activation[-1], y)
		# self.quadratic_cost_derivative(list_activation[-1], y) * \
		# 	sigmoid_derivative(list_activation[-1]) #the first part is the derivate of quadratic cost function
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, list_activation[-2].transpose())

		for l in range(2, self.nb_layers):
			z = list_z[-l]
			delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_derivative(z)
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, list_activation[-l -1].transpose())
		return (nabla_w, nabla_b)
	
	def mini_batch_gradient(self, training_data, epochs, mini_batch_size, learning_rate,\
			lambda_=0.0, test_data=None, n_epoch_early_stop = 0):
		"""Train the neural network using mini-batch stochastic
		gradient descent.  The "training_data" is a list of tuples
		"(x, y)" representing the training inputs and the desired
		outputs.  The other non-optional parameters are
		self-explanatory.  If "test_data" is provided then the
		network will be evaluated against the test data after each
		epoch, and partial progress printed out.  This is useful for
		tracking progress, but slows things down substantially."""
		
		#EarlyStop Initialize
		best_accuracy = 1
		no_change_best_accuracy = 0

		training_data = list(training_data)
		training_size = len(training_data)
		if test_data:
			test_data = list(test_data)
			test_size = len(test_data)
		for j in range(epochs):
			random.shuffle(training_data)
			for n in range(0, training_size, mini_batch_size):
				self.train_(training_data[n: n + mini_batch_size], learning_rate, lambda_, training_size)
			if test_data:
				accuracy = self.evaluate(test_data)
				print("Epoch {0}: {1} / {2}".format(
					j, accuracy, test_size))
				if n_epoch_early_stop > 0:
					if best_accuracy < accuracy:
						best_accuracy = accuracy
						no_change_best_accuracy = 0
					else:
						no_change_best_accuracy += 1
					if no_change_best_accuracy == n_epoch_early_stop:
						return
			else:
				print("Epoch {0} complete".format(j))
	
	def train_(self, batch, learning_rate, lambda_, n):
		"""
			Update the network's weights and biases by applying
			gradient descent using backpropagation to a single mini batch.
			The "mini_batch" is a list of tuples "(x, y)", and "learning_rate"
			is the learning rate.
			``lambda_`` is the L2 regularization parameter who reduce overfitting,
			and ``n`` is the total size of the training data set.
		"""

		nabla_w = [np.zeros(w.shape) for w in self.weights]
		nabla_b = [np.zeros(b.shape) for b in self.biases]

		for x, y in batch:
			delta_nabla_w, delta_nabla_b = self.backpropagation(x, y)
			nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]		
		self.weights = [(1-(learning_rate* lambda_ /n))*w - (learning_rate / len(batch) * dw) for w, dw in zip(self.weights, nabla_w)]
		self.biases = [b - (learning_rate / len(batch) * db) for b, db in zip(self.biases, nabla_b)]


	def evaluate(self, test_data):
		"""Return the number of test inputs for which the neural
		network outputs the correct result. Note that the neural
		network's output is assumed to be the index of whichever
		neuron in the final layer has the highest activation."""
		test_results = [(np.argmax(self.feedforward(x)), y)
						for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)


# nn = Network([3, 2, 1])
# print(nn.feedforward(np.random.randn(3, 1)), "\n\n")
# print(nn.weights)