Assignment 2 - Two-Layer Neural Network on CIFAR-10 

Code:
* CifarDataset - load and process dataset
* NeuralNetworkLayer - architecture of neural network
* ForwardPropagate - forward propagation of neural network
* BackwardPropagate - backward propagation of neural network (mainy for computing gradients)
* Optimizer - trains weight and biases in neural network
* main - main code

How to run:
1. Change root_path in main.py to root path fro cifar-10-batches-py dataset.
2. Change lambda, batch size, epochs and learning rate in main.py when calling the train_mini_batch_GD() method for Optimizer object.