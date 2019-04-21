from CifarDataset import CifarDataset
from Optimizer import Optimizer
from NeuralNetworkLayers import NeuralNetworkLayers

if __name__ == '__main__':

	# Path to dataset root folder
	root_path = "/Users/pedramfk/Workspace/KTH/DD2424/assignment1/code/cifar-10-batches-py/"
	
	# Dataset
	dataset = CifarDataset(root_path, load_all_data = True, N_val = 5000)

	# Neural network framework
	HIDDEN_NODES = 100
	layers = NeuralNetworkLayers(dataset, m = HIDDEN_NODES)

	# Train neural network
	optimizer = Optimizer(dataset, layers)
	optimizer.train_mini_batch_GD(lamb = 0.0005, batch_size = 100, N_epochs = 30, eta = 0.01, eta_step = 2333)

	# Plot results
	optimizer.plot_accuracy()
	optimizer.plot_loss()
	optimizer.plot_cost()
	optimizer.plot_etas()