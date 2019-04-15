import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle

class CifarDataset:

	__N_images = 10000 # number of images
	__d = 3072 # dimensionality of each image
	__K = 10 # number of labels

	def __init__(self, root_path="/Users/pedramfk/Workspace/KTH/DD2424/assignment1/code/cifar-10-batches-py/"):

		# Raw data
		A_train = CifarDataset.__unpickle(root_path + "data_batch_1")
		A_val = CifarDataset.__unpickle(root_path + "data_batch_2")
		A_test = CifarDataset.__unpickle(root_path + "test_batch")

		# Pre-processed data
		self.__X_train = CifarDataset.__get_X(A_train)
		self.__Y_train = CifarDataset.__get_Y(A_train)

		self.__X_val = CifarDataset.__get_X(A_val)
		self.__Y_val = CifarDataset.__get_Y(A_val)

		self.__X_test = CifarDataset.__get_X(A_test)
		self.__Y_test = CifarDataset.__get_Y(A_test)

	@staticmethod
	def __unpickle(file):
		with open(file, 'rb') as fo:
			dict = pickle.load(fo, encoding = 'latin1')
		return dict	

	@staticmethod
	def __get_X(A):
		return np.transpose( A.get('data') ) / 255.0

	@staticmethod
	def __get_Y(A):

		Y = np.array( A.get('labels') ).reshape((-1, 1))
		enc = OneHotEncoder(handle_unknown = 'ignore')
		enc.fit(Y)
	
		return np.transpose( enc.transform(Y).toarray() )

	'''
		Parameters in dataset.
	'''

	def number_of_images(self):
		return self.__N_images

	def dimensionality(self):
		return self.__d

	def number_of_labels(self):
		return self.__K

	'''
		Train/validation/test data.
	'''

	def X_train(self):
		return self.__X_train

	def Y_train(self):
		return self.__Y_train


	def X_val(self):
		return self.__X_val

	def Y_val(self):
		return self.__Y_val


	def X_test(self):
		return self.__X_test

	def Y_test(self):
		return self.__Y_test


#root_path = "/Users/pedramfk/Workspace/KTH/DD2424/assignment1/code/cifar-10-batches-py/"

#dataset = CifarDataset(root_path)
#print( dataset.X_train().shape )




