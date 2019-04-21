from CifarDataset import CifarDataset
from NeuralNetworkLayers import NeuralNetworkLayers
from ForwardPropagate import ForwardPropagate
from BackwardPropagate import BackwardPropagate

import numpy as np
import matplotlib.pyplot as plt

class Optimizer:

	dataset = None
	layers = None

	forwardPropagate = None
	backwardPropagate = None

	__train_cost = None
	__train_loss = None
	__train_acc = None

	__val_cost = None
	__val_loss = None
	__val_acc = None

	__etas = None

	def __init__(self, dataset: CifarDataset, layers: NeuralNetworkLayers):

		self.dataset = dataset
		self.layers = layers

		self.forwardPropagate = ForwardPropagate(self.layers)
		self.backwardPropagate = BackwardPropagate(self.layers)

	@staticmethod
	def get_cross_entropy(Y, P):

		a = np.multiply(Y, P)
		a = a.sum(axis=0)

		a[a == 0] = np.finfo(float).eps
	
		return - np.log(a)

	def compute_loss(self, X, Y):

		s1, h, P = self.forwardPropagate.get_softmax_probabilities(X)

		cross_entropy = Optimizer.get_cross_entropy(Y, P)
		loss = np.mean( cross_entropy )

		return loss

	def compute_cost(self, X, Y, lamb):

		loss = self.compute_loss(X, Y)
		regularization = lamb * ( np.sum( np.power(self.layers.W1, 2) ) + np.sum( np.power(self.layers.W2, 2) ) )
		cost = loss + regularization

		return cost

	def get_accuracy(self, X, Y):

		s1, h, P = self.forwardPropagate.get_softmax_probabilities(X)

		y_predicted = np.argmax(P, axis = 0)
		y_truth = np.argmax(Y, axis = 0)

		N = y_predicted.size
		acc = N - np.count_nonzero( np.subtract(y_truth, y_predicted) )

		return float(acc)/float(N)

	@staticmethod
	def __get_eta_triangular(iterations, eta_min=1e-5, eta_max=1e-1, eta_step=500):

		cycle = np.floor( 1.0 + ( float(iterations) ) / ( 2.0 * float(eta_step) ) )

		X = np.abs( float(iterations) / float(eta_step) - 2.0 * cycle + 1.0 )

		eta_t = eta_min + ( eta_max - eta_min ) * np.maximum(0, (1.0 - X))

		return eta_t

	def __update_weights(self, X_train_batch, Y_train_batch, lamb, eta):

		s1, h, P = self.forwardPropagate.get_softmax_probabilities(X_train_batch)
		delta_J_W1, delta_J_b1, delta_J_W2, delta_J_b2 = self.backwardPropagate.get_cost_gradients(X_train_batch, Y_train_batch, s1, h, P, lamb)

		self.layers.W1 = self.layers.W1 - eta * delta_J_W1
		self.layers.W2 = self.layers.W2 - eta * delta_J_W2
		self.layers.b1 = self.layers.b1 - eta * delta_J_b1
		self.layers.b2 = self.layers.b2 - eta * delta_J_b2

	def train_mini_batch_GD(self, lamb = 0.1, batch_size = 1000, N_epochs = 40, eta = 0.01, eta_step = 500):

		N = self.dataset.number_of_images() # number of images

		N_batches = int( N / batch_size ) # number of batches

		N_iterations = N_epochs * N_batches # number of iterations

		N_samples = N_epochs

		self.__train_cost = np.zeros( N_samples )
		self.__train_loss = np.zeros( N_samples )
		self.__train_acc = np.zeros( N_samples )

		self.__val_cost = np.zeros( N_samples )
		self.__val_loss = np.zeros( N_samples )
		self.__val_acc = np.zeros( N_samples )

		self.__etas = np.zeros( N_iterations )

		iteration = -1
		for epoch in range(N_epochs):
			for batch in range(N_batches):

				iteration += 1

				# Select batch indexes
				index_start = batch * batch_size
				index_end = ( batch + 1 ) * batch_size

				# Select batch data
				X_train_batch = self.dataset.X_train()[:, index_start:index_end]
				Y_train_batch = self.dataset.Y_train()[:, index_start:index_end]

				# Get dynamic eta
				eta = Optimizer.__get_eta_triangular(iteration, eta_step = eta_step)
				self.__etas[iteration] = eta

				# Perform gradient on batch data and update W, b
				self.__update_weights(X_train_batch, Y_train_batch, lamb, eta)

			print("--- epoch finished: " + str( epoch + 1 ) + " --- iterations: " + str( iteration + 1 ))

			# Train/Validation loss/cost/accuracy
			self.__train_loss[epoch] = self.compute_loss(self.dataset.X_train(), self.dataset.Y_train())
			self.__val_loss[epoch] = self.compute_loss(self.dataset.X_val(), self.dataset.Y_val())

			self.__train_cost[epoch] = self.compute_cost(self.dataset.X_train(), self.dataset.Y_train(), lamb)
			self.__val_cost[epoch] = self.compute_cost(self.dataset.X_val(), self.dataset.Y_val(), lamb)

			self.__train_acc[epoch] = self.get_accuracy(self.dataset.X_train(), self.dataset.Y_train())
			self.__val_acc[epoch] = self.get_accuracy(self.dataset.X_val(), self.dataset.Y_val())

		test_acc = self.get_accuracy(self.dataset.X_test(), self.dataset.Y_test())
		print("\n*** Final test accuracy: " + str(test_acc) + " ***\n")

	def plot_accuracy(self):

		t = np.arange(0, self.__train_acc.size)
		
		plt.plot(t, self.__train_acc, self.__val_acc)
		plt.title("Accuracy")
		plt.legend(('Train accuracy', 'Validation accuracy'), loc='upper left')
		plt.grid()
		plt.xlabel("Epoch")
		plt.show()

	def plot_loss(self):

		t = np.arange(0, self.__train_loss.size)
		
		plt.plot(t, self.__train_loss, self.__val_loss)
		plt.title("Loss")
		plt.legend(('Train loss', 'Validation loss'), loc='upper right')
		plt.grid()
		plt.xlabel("Epoch")
		plt.show()

	def plot_cost(self):

		t = np.arange(0, self.__train_cost.size)
	
		plt.plot(t, self.__train_cost, self.__val_cost)
		plt.title("Cost")
		plt.legend(('Train cost', 'Validation cost'), loc='upper right')
		plt.grid()
		plt.xlabel("Epoch")
		plt.show()

	def plot_etas(self):

		t = np.arange(0, self.__etas.size)
		
		plt.plot(t, self.__etas)
		plt.title("Learning Rates")
		plt.grid()
		plt.xlabel("Iteration")
		plt.ylabel("eta")
		plt.show()

if __name__ == '__main__':

	root_path = "/Users/pedramfk/Workspace/KTH/DD2424/assignment1/code/cifar-10-batches-py/"
	dataset = CifarDataset(root_path, load_all_data = True, N_val = 5000)

	HIDDEN_NODES = 50
	layers = NeuralNetworkLayers(dataset, m = HIDDEN_NODES)

	optimizer = Optimizer(dataset, layers)
	optimizer.train_mini_batch_GD(lamb = 0.0005, batch_size = 100, N_epochs = 40, eta = 0.01, eta_step = 2333)

	optimizer.plot_accuracy()
	optimizer.plot_loss()
	optimizer.plot_cost()
	optimizer.plot_etas()



