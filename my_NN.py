import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
	if x.size == 0:
		return None
	x = x.astype(np.float)
	if x.ndim == 0:
		x = np.array(x, ndmin=1)
	return (1.0 / (1.0 + (np.exp(x * -1))))

def sigmoid_derivative(x):
	return sigmoid(x) * (1.0 - sigmoid(x))

# class Network(object):
# 	''' Exemple of size: [2, 3, 1] 
# 		if we want to create a Network object with 
# 			2 neurons in the first layer, 
# 			3 neurons in the second layer, and 
# 			1 neuron in the final layer
# 	'''
# 	def __init__(self, sizes):
# 		self.num_layers = len(sizes)
# 		self.sizes = sizes
# 		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
# 		self.weights = [np.random.randn(y, x) 
# 						for x, y in zip(sizes[:-1], sizes[1:])]

# 	def feedforward(self, a):
# 		"""Return the output of the network if "a" is input.
# 			a′=σ(wa+b)
# 		"""
# 		for b, w in zip(self.biases, self.weights):
# 			a = sigmoid(np.dot(w, a)+b)
# 		return a

# 	def SGD(self, training_data, epochs, mini_batch_size, learning_rate,
# 			test_data=None):
# 		"""Train the neural network using mini-batch stochastic
# 		gradient descent.  The "training_data" is a list of tuples
# 		"(x, y)" representing the training inputs and the desired
# 		outputs.  The other non-optional parameters are
# 		self-explanatory.  If "test_data" is provided then the
# 		network will be evaluated against the test data after each
# 		epoch, and partial progress printed out.  This is useful for
# 		tracking progress, but slows things down substantially."""
# 		if test_data: n_test = len(test_data)
# 		n = len(training_data)
# 		for j in xrange(epochs):
# 			random.shuffle(training_data)
# 			mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
# 			for mini_batch in mini_batches:
# 				self.train_(mini_batch, learning_rate)
# 			if test_data:
# 				print "Epoch {0}: {1} / {2}".format(
# 					j, self.evaluate(test_data), n_test)
# 			else:
# 				print "Epoch {0} complete".format(j)

# 	def train_(self, mini_batch, learning_rate):
# 		"""Update the network's weights and biases by applying
# 		gradient descent using backpropagation to a single mini batch.
# 		The "mini_batch" is a list of tuples "(x, y)", and "learning_rate"
# 		is the learning rate."""
# 		nabla_b = [np.zeros(b.shape) for b in self.biases]
# 		nabla_w = [np.zeros(w.shape) for w in self.weights]
# 		for x, y in mini_batch:
# 			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
# 			nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
# 			nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
# 		self.weights = [w-(learning_rate/len(mini_batch))*nw 
# 						for w, nw in zip(self.weights, nabla_w)]
# 		self.biases = [b-(learning_rate/len(mini_batch))*nb 
# 					for b, nb in zip(self.biases, nabla_b)]

# 	def backprop(self, x, y):
# 		"""Return a tuple ``(nabla_b, nabla_w)`` representing the
# 		gradient for the cost function C_x.  ``nabla_b`` and
# 		``nabla_w`` are layer-by-layer lists of numpy arrays, similar
# 		to ``self.biases`` and ``self.weights``."""
# 		nabla_b = [np.zeros(b.shape) for b in self.biases]
# 		nabla_w = [np.zeros(w.shape) for w in self.weights]
# 		# feedforward
# 		activation = x
# 		activations = [x] # list to store all the activations, layer by layer
# 		zs = [] # list to store all the z vectors, layer by layer
# 		for b, w in zip(self.biases, self.weights):
# 			z = np.dot(w, activation)+b
# 			zs.append(z)
# 			activation = sigmoid(z)
# 			activations.append(activation)
# 		# backward pass
# 		delta = self.cost_derivative(activations[-1], y) * \
# 			sigmoid_prime(zs[-1])
# 		nabla_b[-1] = delta
# 		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
# 		# Note that the variable l in the loop below is used a little
# 		# differently to the notation in Chapter 2 of the book.  Here,
# 		# l = 1 means the last layer of neurons, l = 2 is the
# 		# second-last layer, and so on.  It's a renumbering of the
# 		# scheme in the book, used here to take advantage of the fact
# 		# that Python can use negative indices in lists.
# 		for l in xrange(2, self.num_layers):
# 			z = zs[-l]
# 			sp = sigmoid_prime(z)
# 			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
# 			nabla_b[-l] = delta
# 			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
# 		return (nabla_b, nabla_w)

# 	def evaluate(self, test_data):
# 		"""Return the number of test inputs for which the neural
# 		network outputs the correct result. Note that the neural
# 		network's output is assumed to be the index of whichever
# 		neuron in the final layer has the highest activation."""
# 		test_results = [(np.argmax(self.feedforward(x)), y)
# 						for (x, y) in test_data]
# 		return sum(int(x == y) for (x, y) in test_results)

# 	def cost_derivative(self, output_activations, y):
# 		"""Return the vector of partial derivatives \partial C_x /
# 		\partial a for the output activations."""
# 		return (output_activations-y)


arr = np.random.randn(3, 2)
sizes = [5, 1, 3]
# print(arr)
wgt = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
print(wgt)