import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle

class CifarDataset:

	__N_images_batch = 10000 # number of images in each batch
	__d = 3072 # dimensionality of each image
	__K = 10 # number of labels

	__load_all_data = False
	__N_val = 5000 # validation data size

	__root_path = None

	def __init__(self, root_path = "/Users/pedramfk/Workspace/KTH/DD2424/assignment1/code/cifar-10-batches-py/", load_all_data = False, N_val = 5000):

		# Set root path
		self.__root_path = root_path

		# If all data should be loaded
		self.__load_all_data = load_all_data

		# Size of validation set
		self.__N_val = N_val

		if load_all_data:
			self.__load_data_all()
		else:
			self.__load_data_partial()


	def __load_data_partial(self):

		# Raw data
		A_train = CifarDataset.__unpickle(self.__root_path + "data_batch_1")
		A_val = CifarDataset.__unpickle(self.__root_path + "data_batch_2")
		A_test = CifarDataset.__unpickle(self.__root_path + "test_batch")

		# Pre-processed data
		self.__X_train = CifarDataset.__get_X(A_train)
		self.__Y_train = CifarDataset.__get_Y(A_train)

		self.__X_val = CifarDataset.__get_X(A_val)
		self.__Y_val = CifarDataset.__get_Y(A_val)

		self.__X_test = CifarDataset.__get_X(A_test)
		self.__Y_test = CifarDataset.__get_Y(A_test)

		self.__X_train = CifarDataset.__center_data(self.__X_train, self.__d)
		self.__X_val = CifarDataset.__center_data(self.__X_val, self.__d)
		self.__X_test = CifarDataset.__center_data(self.__X_test, self.__d)

	def __load_data_all(self):

		N = ( self.__N_images_batch - self.__N_val )

		# Raw data
		A_train1 = CifarDataset.__unpickle(self.__root_path + "data_batch_1")
		A_train2 = CifarDataset.__unpickle(self.__root_path + "data_batch_2")
		A_train3 = CifarDataset.__unpickle(self.__root_path + "data_batch_3")
		A_train4 = CifarDataset.__unpickle(self.__root_path + "data_batch_4")
		A_train5 = CifarDataset.__unpickle(self.__root_path + "data_batch_5") # this will be split into train/val
		A_test = CifarDataset.__unpickle(self.__root_path + "test_batch")

		# Pre-processed data
		self.__X_train = CifarDataset.__get_X(A_train1)
		self.__X_train = np.concatenate( (self.__X_train, CifarDataset.__get_X(A_train2)), axis = 1 )
		self.__X_train = np.concatenate( (self.__X_train, CifarDataset.__get_X(A_train3)), axis = 1 )
		self.__X_train = np.concatenate( (self.__X_train, CifarDataset.__get_X(A_train4)), axis = 1 )
		self.__X_train = np.concatenate( (self.__X_train, CifarDataset.__get_X(A_train5)[:, 0:N] ), axis = 1 )

		self.__Y_train = CifarDataset.__get_Y(A_train1)
		self.__Y_train = np.concatenate( (self.__Y_train, CifarDataset.__get_Y(A_train2)), axis = 1 )
		self.__Y_train = np.concatenate( (self.__Y_train, CifarDataset.__get_Y(A_train3)), axis = 1 )
		self.__Y_train = np.concatenate( (self.__Y_train, CifarDataset.__get_Y(A_train4)), axis = 1 )
		self.__Y_train = np.concatenate( (self.__Y_train, CifarDataset.__get_Y(A_train5)[:, 0:N] ), axis = 1 )

		self.__X_val = CifarDataset.__get_X(A_train5)[:, N:]
		self.__Y_val = CifarDataset.__get_Y(A_train5)[:, N:]

		self.__X_test = CifarDataset.__get_X(A_test)
		self.__Y_test = CifarDataset.__get_Y(A_test)

		self.__X_train = CifarDataset.__center_data(self.__X_train, self.__d)
		self.__X_val = CifarDataset.__center_data(self.__X_val, self.__d)
		self.__X_test = CifarDataset.__center_data(self.__X_test, self.__d)

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

	@staticmethod
	def __center_data(X, d):
	
		X_mean = np.mean(X, axis = 1).reshape((d, 1))
		X_centered = np.subtract(X, X_mean)

		return X_centered

	'''
		Parameters in dataset.
	'''

	def number_of_images(self):
		return self.__X_train.shape[1]

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

#dataset = CifarDataset(root_path, load_all_data = True, N_val = 5000)
#print( dataset.number_of_images() )




