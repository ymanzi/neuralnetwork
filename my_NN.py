import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
	if x.size == 0:
		return None
	x = x.astype(float)
	if x.ndim == 0:
		x = np.array(x, ndmin=1)
	return (1.0 / (1.0 + (np.exp(x * -1))))

def sigmoid_derivative(x):
	return sigmoid(x) * (1.0 - sigmoid(x))

class Network(object):
	def __init__(self, layers):
		''' Exemple of size: [2, 3, 1] 
		if we want to create a Network object with 
			2 neurons in the first layer, 
			3 neurons in the second layer, and 
			1 neuron in the final layer
		'''
		self.layers = layers
		self.nb_layers = len(layers)
		self.weights = [np.random.randn(x, y) for x,y in zip(layers[1:], layers[:-1])]
		self.biases = [np.random.randn(x, 1) for x in layers[1:]]

	def quadratic_cost_derivative(self, output_activations, y):
		np.subtract(output_activations, y)

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
		for weight, bias in zip(self.weights, self.biases):
			z = np.add(np.dot(weight, a), bias)
			a = sigmoid(z)
			list_activation.append(a)
			list_z.append(z)
		delta = self.quadratic_cost_derivative(list_activation[-1], y) * sigmoid_derivative(list_activation[-1]) #the first part is the derivate of quadratic cost function
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, list_activation[-2].transpose())

		for l in range(2, self.nb_layers):
			z = list_z[-l]
			delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_derivative(z)
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, list_activation[-l -1].transpose())
		return (nabla_w, nabla_b)
	
	def mini_batch_gradient(self, training_data, epochs, mini_batch_size, learning_rate,
			test_data=None):
		"""Train the neural network using mini-batch stochastic
		gradient descent.  The "training_data" is a list of tuples
		"(x, y)" representing the training inputs and the desired
		outputs.  The other non-optional parameters are
		self-explanatory.  If "test_data" is provided then the
		network will be evaluated against the test data after each
		epoch, and partial progress printed out.  This is useful for
		tracking progress, but slows things down substantially."""
		
		# training_size = len(training_data)
		if test_data:
			test_size = len(test_data)
		training_data = np.ndarray(training_data)
		np.random.shuffle(training_data)
		start = 0
		for j in range(epochs):
			if start + 32 > training_data.shape[0]:
				start = 0
				np.random.shuffle(training_data)
			self.train_(training_data[start: start + mini_batch_size], learning_rate)
			if test_data:
				msg = "Epoch {0}: {1} / {2}".format(
					j, self.evaluate(test_data), test_size)
			else:
				msg = "Epoch {0} complete".format(j)
			print(msg)
	
	def train_(self, data, learning_rate):
		"""
			Update the network's weights and biases by applying
			gradient descent using backpropagation to a single mini batch.
			The "mini_batch" is a list of tuples "(x, y)", and "learning_rate"
			is the learning rate.
		"""

		nabla_w = [np.zeros(w.shape) for w in self.weights]
		nabla_b = [np.zeros(b.shape) for b in self.biases]

		for x, y in data:
			delta_nabla_w, delta_nabla_b = self.backpropagation(x, y)
			nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]		
		self.weights = [w - learning_rate / len(data) * dw for w, dw in zip(self.weights, nabla_w)]
		self.biases = [b - learning_rate / len(data) * db for b, db in zip(self.biases, nabla_b)]


	def evaluate(self, test_data):
		"""Return the number of test inputs for which the neural
		network outputs the correct result. Note that the neural
		network's output is assumed to be the index of whichever
		neuron in the final layer has the highest activation."""
		test_results = [(np.argmax(self.feedforward(x)), y)
						for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)


nn = Network([3, 2, 1])
print(nn.feedforward(np.random.randn(3, 1)), "\n\n")
# print(nn.weights)