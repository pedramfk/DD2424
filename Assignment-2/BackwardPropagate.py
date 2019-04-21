from NeuralNetworkLayers import NeuralNetworkLayers

import numpy as np

class BackwardPropagate:

	layers = None

	def __init__(self, layers: NeuralNetworkLayers):

		self.layers = layers


	@staticmethod
	def indicator_function(s):

		s[ np.where( s > 0 ) ] = 1
		s[ np.where( s <= 0 ) ] = 0

		return s

	@staticmethod
	def diag(s1):

		s1 = s1[0:s1.shape[1], :]
		s1_transformed = np.zeros((s1.shape[0], s1.shape[0]))

		for i in range(s1.shape[0]):

			if s1[i, i] > 0:
				s1_transformed[i, i] = 1
			else:
				s1_transformed[i, i] = 0

		return s1_transformed

	def get_L_gradients(self, X, Y, s1, h, P):

		n_b = X.shape[1] # batch size

		#s1, h, P = get_P(W1, b1, W2, b2, X)

		G = - np.subtract(Y, P)

		delta_L_W2 = np.matmul(G, h.T) / n_b
		delta_L_b2 = np.matmul(G, np.ones((n_b, 1))) / n_b

		G = np.matmul(self.layers.W2.T, G)
		h[ h > 0 ] = 1
		h[ h <= 0 ] = 0
		G = G * h

		delta_L_W1 = np.matmul(G, X.T) / n_b
		delta_L_b1 = np.matmul(G, np.ones((n_b, 1)))

		return delta_L_W1, delta_L_b1, delta_L_W2, delta_L_b2

	def get_cost_gradients(self, X, Y, s1, h, P, lamb):

		delta_L_W1, delta_L_b1, delta_L_W2, delta_L_b2 = self.get_L_gradients(X, Y, s1, h, P)

		delta_J_W1 = delta_L_W1 + 2 * lamb * self.layers.W1
		delta_J_W2 = delta_L_W2 + 2 * lamb * self.layers.W2

		delta_J_b1 = delta_L_b1
		delta_J_b2 = delta_L_b2

		return delta_J_W1, delta_J_b1, delta_J_W2, delta_J_b2


