from CifarDataset import CifarDataset
from NeuralNetworkLayer import NeuralNetworkLayer

import numpy as np

class Optimizer(NeuralNetworkLayer):

	def __init__(self, dataset: CifarDataset):

		super().__init__(dataset)


	def update_weights(self, X_train_batch, Y_train_batch, lamb, eta):

		delta_J_W, delta_J_b = self.get_cost_gradients(X_train_batch, Y_train_batch, lamb)
			
		self.W = self.W - eta * delta_J_W
		self.b = self.b - eta * delta_J_b


	# Mini Batch Gradient Descent
	def train(self, lamb=0.1, n_batch=1000, n_epochs=40, eta=0.01):

		N = self.dataset.number_of_images()

		train_loss = np.zeros( n_epochs )
		val_loss = np.zeros( n_epochs )

		train_cost = np.zeros( n_epochs )
		val_cost = np.zeros( n_epochs )

		train_acc = np.zeros( n_epochs )
		val_acc = np.zeros( n_epochs )

		for i in range(n_epochs):

			for j in range(0, int(N/n_batch)):

				# Select batch indexes
				j_start = j*n_batch
				j_end = (j+1)*n_batch

				# Select batch data
				X_train_batch = self.dataset.X_train()[:, j_start:j_end]
				Y_train_batch = self.dataset.Y_train()[:, j_start:j_end]

				# Perform gradient on batch data and update W, b
				self.update_weights(X_train_batch, Y_train_batch, lamb, eta)

			print( "epoch: " + str(i + 1) )

			# Train/Validation loss for this epoch
			train_loss[i] = self.compute_loss(self.dataset.X_train(), self.dataset.Y_train())
			val_loss[i] = self.compute_loss(self.dataset.X_val(), self.dataset.Y_val())

			# Train/Validation cost for this epoch
			train_cost[i] = self.compute_cost(self.dataset.X_train(), self.dataset.Y_train(), lamb)
			val_cost[i] = self.compute_cost(self.dataset.X_val(), self.dataset.Y_val(), lamb)

			# Train/Validation loss/cost/accuracy for this epoch
			train_acc[i] = self.get_accuracy(self.dataset.X_train(), self.dataset.Y_train())
			val_acc[i] = self.get_accuracy(self.dataset.X_val(), self.dataset.Y_val())

		test_acc = self.get_accuracy(self.dataset.X_test(), self.dataset.Y_test())
		print("Final test accuracy: " + str(test_acc))

		return train_loss, val_loss, train_cost, val_cost, train_acc, val_acc, test_acc


if __name__ == '__main__':

	root_path = "/Users/pedramfk/Workspace/KTH/DD2424/assignment1/code/cifar-10-batches-py/"
	dataset = CifarDataset(root_path)

	optimizer = Optimizer(dataset)

	train_loss, val_loss, train_cost, val_cost, train_acc, val_acc, test_acc = optimizer.train()

	print(test_acc)




