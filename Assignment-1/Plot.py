import matplotlib.pyplot as plt
import numpy as np

class Plot:

	@staticmethod
	def images(W):

		for i in range(10):

			plt.subplot(2, 5, i+1)

			row = W[i, :]
			img = (row - row.min()) / (row.max() - row.min())
			squared_image = np.rot90(np.reshape(img, (32, 32, 3), order='F'), k=3)
	
			plt.imshow(squared_image, interpolation='gaussian')
			plt.axis('off')
		
			title = "Image {}".format(i)
			plt.title(title)

		plt.show()

	@staticmethod
	def accuracy(train_acc, val_acc):

		t = np.arange(0, train_acc.size)
		
		plt.plot(t, train_acc, val_acc)
		plt.title("Accuracy")
		plt.legend(('Train accuracy', 'Validation accuracy'), loc='upper left')
		plt.grid()
		plt.xlabel("Epoch")
		plt.show()

	@staticmethod
	def loss(train_loss, val_loss):

		t = np.arange(0, train_loss.size)
		
		plt.plot(t, train_loss, val_loss)
		plt.title("Loss")
		plt.legend(('Train loss', 'Validation loss'), loc='upper right')
		plt.grid()
		plt.xlabel("Epoch")
		plt.show()

	@staticmethod
	def cost(train_cost, val_cost):

		t = np.arange(0, train_cost.size)
	
		plt.plot(t, train_cost, val_cost)
		plt.title("Cost")
		plt.legend(('Train cost', 'Validation cost'), loc='upper right')
		plt.grid()
		plt.xlabel("Epoch")
		plt.show()


