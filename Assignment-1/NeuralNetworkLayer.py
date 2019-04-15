from CifarDataset import CifarDataset

import numpy as np

class NeuralNetworkLayer:

	# Node weights and biases.
	W = None
	b = None

	__N_images = None # number of images
	__d = None # dimensionality of each image
	__K = None # number of labels

	# Dataset
	dataset = None

	def __init__(self, dataset: CifarDataset):

		self.dataset = dataset
		self.__N_images = dataset.number_of_images()
		self.__d = dataset.dimensionality()
		self.__K = dataset.number_of_labels()

		self.__initialize_weight_and_bias()


	def __initialize_weight_and_bias(self, mean=0.0, sigma=0.01):

		self.W = np.random.normal(mean, sigma, [self.__K, self.__d])
		self.b = np.random.normal(mean, sigma, [self.__K, 1])


	def get_number_of_images(self):
		return self.__N_images

	def get_input_dimensionality(self):
		return self.__d

	def get_number_of_labels(self):
		return self.__K

	'''
		Network functions.
	'''

	def get_softmax_probabilities(self, X):

		s = np.matmul(self.W, X) + self.b
		s_exp = np.exp( s )
		
		return s_exp / np.sum(s_exp, axis=0)

	def get_cross_entropy(self, Y, P):

		a = np.multiply(Y, P)
		a = a.sum(axis=0)
	
		return -np.log(a)

	def compute_loss(self, X, Y):

		P = self.get_softmax_probabilities(X)

		batch_size = X.shape[1]
		cross_entropy = self.get_cross_entropy(Y, P)

		loss = np.mean(cross_entropy)

		return loss

	def compute_cost(self, X, Y, lamb):

		regularization_cost = lamb * np.square(self.W).sum()
		cost = self.compute_loss(X, Y) + regularization_cost

		return cost

	def get_accuracy(self, X, Y):

		P = self.get_softmax_probabilities(X)

		y_predicted = np.argmax(P, axis = 0)
		y_truth = np.argmax(Y, axis = 0)

		N = y_predicted.size
		acc = N - np.count_nonzero( np.subtract(y_truth, y_predicted) )

		return float(acc)/float(N)

	def get_gradients(self, X, Y):

		P = self.get_softmax_probabilities(X)
		G = - ( np.subtract(Y, P) )

		batch_size = X.shape[1]

		delta_L_W = np.matmul(G, X.T) / batch_size

		one = np.ones((batch_size))
	
		delta_L_b = ( np.matmul(G, one) / batch_size ).reshape((-1, 1))

		return delta_L_W, delta_L_b

	def get_cost_gradients(self, X, Y, lamb):

		delta_L_W, delta_L_b = self.get_gradients(X, Y)

		delta_J_W = delta_L_W + 2*lamb*self.W
		delta_J_b = delta_L_b

		return delta_J_W, delta_J_b

