from CifarDataset import CifarDataset
from Optimizer import Optimizer
from Plot import Plot

if __name__ == '__main__':

	# Path to dataset root folder
	root_path = "/Users/pedramfk/Workspace/KTH/DD2424/assignment1/code/cifar-10-batches-py/"
	
	# Dataset
	dataset = CifarDataset(root_path)

	# Train neural network
	optimizer = Optimizer(dataset)
	train_loss, val_loss, train_cost, val_cost, train_acc, val_acc, test_acc = optimizer.train(lamb=0.3, n_batch=1000, n_epochs=80, eta=0.01)

	# Plot results
	Plot.images(optimizer.W)
	Plot.accuracy(train_acc, val_acc)
	Plot.loss(train_loss, val_loss)
	Plot.cost(train_cost, val_cost)