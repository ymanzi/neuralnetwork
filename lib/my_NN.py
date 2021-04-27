import numpy as np
import random
import matplotlib.pyplot as plt
from lib.data_init import init_array
from lib.metrics import *
from lib.activation_functions import *
from lib.cost_functions import *
from lib.weight_init import *


		# return 1 * ReLu.fct(x)
		# return np.maximum(0.01, ReLu.fct(x))

# def sigmoid(x: np.ndarray) -> np.ndarray:
# 	if x.size == 0:
# 		return None
# 	x = x.astype(float)
# 	if x.ndim == 0:
# 		x = np.array(x, ndmin=1)
# 	return (1.0 / (1.0 + (np.exp(-x))))

# def sigmoid_derivative(x):
# 	sig = sigmoid(x)
# 	return sig * (1.0 - sig)


def ask_function(question):
	reply = "lol"
	while reply not in ['y', 'n']:
		print("------------------------------------------------------")
		reply = str(input(question + " (y/n): "))
		if reply not in ['y', 'n']:
			print("The only accepted replies are 'y' or 'n'. ")
	return reply


class Network(object):
	def __init__(self, name, layers, cost=CrossEntropyCost, hidden_activation=Sigmoid, output_activation=Sigmoid, w_init='std',\
			epochs=1000, batch_size=32, learning_rate = 1.0, lambda_=0.0, n_epoch_early_stop = 0, momentum=0.0, dropout=1.0):
		''' 
			Exemple of layers: [2, 3, 1] 
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
		self.name = name
		self.layers = layers
		self.nb_layers = len(layers)
		if w_init == 'std':
			self.weights = Weight_init.std(layers)
		elif w_init == 'xavier':
			self.weights = Weight_init.xavier(layers)
		elif w_init == 'he':
			self.weights = Weight_init.he(layers)
		# self.biases = [np.random.randn(x, 1) for x in layers[1:]]
		self.biases = [np.zeros((x, 1)) for x in layers[1:]]
		self.cost = cost
		self.list_train_cost = [[],[]]
		self.list_test_cost = [[],[]]
		self.output_a = output_activation
		self.hidden_a = hidden_activation
		self.epochs = epochs
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.lambda_ = lambda_
		self.n_epoch_early_stop = n_epoch_early_stop
		self.saved_weights = None
		self.saved_biases = None
		self.old_nabla_w = [np.zeros(w.shape) for w in self.weights]
		self.old_nabla_b = [np.zeros(b.shape) for b in self.biases]
		self.momentum_ = momentum
		self.dropout = dropout

	def lunch_test(self, train_data, test_data, val_data):
		def init_(question ,dataset):
			reply = ask_function(question)
			if reply == 'y':
				print("------------------------------------------------------")
				print("                       DATA METRICS                   ")
				print("------------------------------------------------------")
				print("Cross Entropy Cost: {}".format(self.get_cost(dataset)))
				tuple_a_y = list(zip(*[(np.argmax(self.feedforward(x)), np.argmax(y))
						for (x, y) in dataset]))
				predicted = np.array(tuple_a_y[0])
				expected = np.array(tuple_a_y[1])
				print("Accuracy: ", accuracy_score_(predicted, expected))
				print("Precision (check False Positive): ", precision_score_(predicted, expected, 1))
				print("Recall Score (check False Negative): ", recall_score_(predicted, expected, 1))
				print("F1 Score (Both FP and FN): ", f1_score_(predicted, expected, 1))
				print("\nConfusion Matrix:")
				confusion_matrix(predicted, expected, 1)

		if train_data:
			init_("Do you want to test on the train_data ?", train_data)
		if test_data:
			init_("Do you want to test on the test_data ?", test_data)
		if val_data:
			val_data = list(val_data)
			init_("Do you want to test on the validation_data ?", val_data)


	def draw_plot(self):
		reply = ask_function("Do you want to see the graph ?")
		if reply == 'y':
			train0 = list(zip(*self.list_train_cost[0]))
			train1 = list(zip(*self.list_train_cost[1]))
			test0 = list(zip(*self.list_test_cost[0]))
			test1 = list(zip(*self.list_test_cost[1]))
			plt.plot(test0[0], test0[1], label= self.name + ' Test Before Early-Stop')
			plt.plot(train0[0], train0[1], label=self.name + ' Train Before Early-Stop')
			plt.plot(test1[0], test1[1], label=self.name + ' Test After Early-Stop')
			plt.plot(train1[0], train1[1], label=self.name + ' Train After Early-Stop')
			plt.xlabel("Epoch")
			plt.ylabel("Cost")
			title = "Start Cost : {}\n End Cost: {}".format(train0[1][0], train1[1][-1])
			plt.title(title)
			plt.legend()
			plt.show()

	def feedforward(self, a):
		"""Return the output of the network if "a" is input.
			a′=σ(wa+b)
		"""
		for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
			# a = sigmoid(np.add(np.dot(weight, a), bias))
			a = self.hidden_a.fct(np.add(np.dot(weight, a), bias))
		a = self.output_a.fct(np.add(np.dot(self.weights[-1], a), self.biases[-1]))
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

		#dropoutConnect
		dropWeights = [np.random.binomial(1, self.dropout, size=w.shape)/self.dropout for w in self.weights[:-1]]
		dropWeights = [w * d for w,d in zip(self.weights[:-1], dropWeights)]
		dropWeights.append(self.weights[-1])

		dropBiases = [np.random.binomial(1, 1.0, size=b.shape)/self.dropout for b in self.biases[:-1]]
		dropBiases = [b * d for b,d in zip(self.biases[:-1], dropBiases)]
		dropBiases.append(self.biases[-1])

		#feedforward
		list_activation = [x]
		list_z = []
		a = x
		# for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
		for weight, bias in zip(dropWeights, dropBiases[:-1]):
			z = np.add(np.dot(weight, a), bias)
			a = self.hidden_a.fct(z)
			list_activation.append(a)
			list_z.append(z)
		# z = np.add(np.dot(self.weights[-1] , a), self.biases[-1])
		z = np.add(np.dot(dropWeights[-1] , a), dropBiases[-1])
		a = self.output_a.fct(z)
		list_activation.append(a)
		list_z.append(z)

		delta = self.cost.delta(list_z[-1], list_activation[-1], y)
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, list_activation[-2].transpose())
		for l in range(2, self.nb_layers):
			z = list_z[-l]
			# delta = np.dot(self.weights[-l + 1].transpose(), delta) * self.hidden_a.derivative(z)
			delta = np.dot(dropWeights[-l + 1].transpose(), delta) * self.hidden_a.derivative(z)
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, list_activation[-l -1].transpose())
		return (nabla_w, nabla_b)
	
	def train_(self, training_data, test_data=None, validation_data=None):
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
		best_test_cost = 1
		best_train_cost = 1
		prev_train_cost = 0
		prev_test_cost = 0

		diff_cost = -1
		best_diff = 1
		no_change_diff_cost = 0
		no_change = 0

		training_data = list(training_data)
		training_size = len(training_data)
		if test_data:
			test_data = list(test_data)
			test_size = len(test_data)
		for j in range(self.epochs):
			np.random.shuffle(training_data)
			for n in range(0, training_size, self.batch_size):
				if self.batch_size == 1:
					np.random.shuffle(training_data)
				self.update_minibatch(training_data[n: n + self.batch_size], self.learning_rate, self.lambda_, training_size)
			if test_data:
				accuracy = self.evaluate(test_data)
				test_cost = self.get_cost(test_data)
				train_cost = self.get_cost(training_data)
				self.list_test_cost[0].append(test_cost)
				self.list_train_cost[0].append(train_cost)
				
				print("Epoch {}: {} / {} -> Cost: {}  Test Cost: {}  learning_rate: {}".format(
					j, accuracy, test_size, train_cost, test_cost, self.learning_rate))
				if self.n_epoch_early_stop > 0:
					if test_cost < 0.07 and train_cost < 0.07 and np.absolute(test_cost - train_cost) < best_diff\
							and test_cost < best_test_cost:
						best_diff = np.absolute(test_cost - train_cost)
						no_change_diff_cost = 0
						self.saved_biases = self.biases
						self.saved_weights = self.weights
						best_test_cost = test_cost
					elif (test_cost > 0.07 or train_cost > 0.07) and test_cost < best_test_cost:
						best_test_cost = test_cost
						self.saved_biases = self.biases
						self.saved_weights = self.weights
						no_change_diff_cost = 0
					else:
						no_change_diff_cost += 1
					prev_test_cost = test_cost
					prev_train_cost = train_cost
					if no_change_diff_cost == self.n_epoch_early_stop:
						print("Early stop activated")
						self.weights = self.saved_weights
						self.biases = self.saved_biases
						break
			else:
				print("Epoch {0} complete".format(j))
		if test_data:
			tmp_test = list(enumerate(self.list_test_cost[0]))
			tmp_train = list(enumerate(self.list_train_cost[0]))

			#Separate the list cost to be able to draw plot before and after early-stop is activated"
			if self.n_epoch_early_stop:
				self.list_test_cost = [tmp_test[:-self.n_epoch_early_stop + 1], tmp_test[-self.n_epoch_early_stop:]]
				self.list_train_cost = [tmp_train[:-self.n_epoch_early_stop + 1], tmp_train[-self.n_epoch_early_stop:]]
			else:
				self.list_test_cost[0] = tmp_test
				self.list_train_cost[0] = tmp_train
			self.draw_plot()
			self.lunch_test(training_data, test_data, validation_data)

	
	def update_minibatch(self, batch, learning_rate, lambda_, n):
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
		self.weights = [(1-(learning_rate* lambda_ /n))*w - (learning_rate / len(batch) * (dw + self.momentum_ * odw)) for w, dw, odw in zip(self.weights, nabla_w, self.old_nabla_w)]
		self.biases = [b - (learning_rate / len(batch) * (db + self.momentum_ * odb)) for b, db, odb in zip(self.biases, nabla_b, self.old_nabla_b)]
		
		# Momentum
		self.old_nabla_w = nabla_w
		self.old_nabla_b = nabla_b

	def evaluate(self, test_data):
		"""Return the number of test inputs for which the neural
		network outputs the correct result. Note that the neural
		network's output is assumed to be the index of whichever
		neuron in the final layer has the highest activation."""
		test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
						for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)

	def get_cost(self, test_data):
		"""Return the cost. Note that the neural
		network's output is assumed to be the index of whichever
		neuron in the final layer has the highest activation."""

		tmp_res = [(self.feedforward(x), y) for (x,y) in test_data]
		tuple_a_y = list(zip(*[(x, y)	for (x, y) in tmp_res]))
		a = np.array(tuple_a_y[0])
		y = np.array(tuple_a_y[1])

		return self.cost.value(a , y, self.weights, self.lambda_)

# nn = Network([3, 2, 1])
# print(nn.feedforward(np.random.randn(3, 1)), "\n\n")
# print(nn.weights)