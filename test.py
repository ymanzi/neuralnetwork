import numpy as np
import mnist_loader
import my_NN

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = my_NN.Network([784, 30, 10])
net.mini_batch_gradient(training_data, 30, 10, 0.5, test_data=test_data, lambda_=5.0, n_epoch_early_stop=10)