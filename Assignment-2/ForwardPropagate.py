#from CifarDataset import CifarDataset
from NeuralNetworkLayers import NeuralNetworkLayers

import numpy as np

class ForwardPropagate:

	layers = None

	def __init__(self, layers: NeuralNetworkLayers):

		self.layers = layers

	@staticmethod
	def ReLu(s):	
		return s.clip( min = 0 )

	def get_softmax_probabilities(self, X):

		s1 = np.matmul(self.layers.W1, X) + self.layers.b1
		h = ForwardPropagate.ReLu(s1)

		s = np.matmul(self.layers.W2, h) + self.layers.b2
		s_exp = np.exp(s)
		P = s_exp / np.sum(s_exp, axis = 0)
	
		return s1, h, P

