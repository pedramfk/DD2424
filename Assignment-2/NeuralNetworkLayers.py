from CifarDataset import CifarDataset

import numpy as np

class NeuralNetworkLayers:

	# Node weights and biases.
	W1 = None
	b1 = None
	W2 = None
	b2 = None

	__N_images = None # number of images
	__d = None # dimensionality of each image
	__K = None # number of labels
	__m = None # number of nodes in network

	# Dataset
	dataset = None

	def __init__(self, dataset: CifarDataset, m):

		self.dataset = dataset
		self.__m = m
		self.__N_images = dataset.number_of_images()
		self.__d = dataset.dimensionality()
		self.__K = dataset.number_of_labels()

		self.__initialize_weight_and_bias()


	def __initialize_weight_and_bias(self, mean=0.0, sigma=0.01):

		self.W1 = np.random.normal(mean, sigma, [self.__m, self.__d])
		self.b1 = np.random.normal(mean, sigma, [self.__m, 1])

		self.W2 = np.random.normal(mean, sigma, [self.__K, self.__m])
		self.b2 = np.random.normal(mean, sigma, [self.__K, 1])

	def get_number_of_images(self):
		return self.__N_images

	def get_input_dimensionality(self):
		return self.__d

	def get_number_of_labels(self):
		return self.__K


