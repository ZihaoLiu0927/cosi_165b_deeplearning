import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle as pkl
import lr_utils as U


# PA-II-5 write model function 
def model(x_train, y_train, x_test, y_test, num_iter, learning_rate, print_loss):
	# initialize parameters with gaussian_normal (1 line of code)
	w, b = U.initialize_zero(x_train.shape[1])

	# model optimization (1 line of code)
	parameters, loss_all = U.optimize(w, b, x_train, y_train, num_iter, learning_rate, print_loss)

	# load trained parameters w and b 
	w = parameters["w"]
	b = parameters["b"]

	# predict test/train data samples (2 lines of code)
	y_predict_test = U.predict(w, b, x_test)
	y_predict_train = U.predict(w, b, x_train)

	# compute train/test accuracy 
	true_pos_test = len(y_test[np.logical_and(y_predict_test == 1,  y_test == 1)])
	true_neg_test = len(y_test[np.logical_and(y_predict_test == 0,  y_test == 0)])
	accuracy_test = (true_pos_test+true_neg_test) / y_test.shape[1]
	print(f'Predict accurary of the test data after {num_iter} iterations is: {accuracy_test}')
	
	true_pos_train = len(y_train[np.logical_and(y_predict_train == 1,  y_train == 1)])
	true_neg_train = len(y_train[np.logical_and(y_predict_train == 0,  y_train == 0)])
	accuracy_train = (true_pos_train+true_neg_train) / y_train.shape[1]
	print(f'Predict accurary of the train data after {num_iter} iterations with learning rate {learning_rate} is: {accuracy_train}')

	# save result
	result = {"loss_all": loss_all,
		 "y_prediction_test": y_predict_test, 
		 "y_prediction_train" : y_predict_train, 
		 "w" : w, 
		 "b" : b}

	return result



if __name__ == '__main__':
	# load datasets
	train_data_x, train_data_y, test_data_x, test_data_y = U.load_dataset()

	# QI-4 reshape train_data_x/test_data_x as 2D arrays (2 lines of code)
	train_data_x = np.reshape(train_data_x, (train_data_x.shape[0], train_data_x.shape[1] * train_data_x.shape[2] * train_data_x.shape[3]))

	test_data_x = np.reshape(test_data_x, (test_data_x.shape[0], test_data_x.shape[1] * test_data_x.shape[2] * test_data_x.shape[3]))

	# normalize data 
	train_data_x = train_data_x / 255.0
	test_data_x = test_data_x / 255.0
 
	# run model 
	result = model(train_data_x, train_data_y, test_data_x, test_data_y, 5000, 0.01, True)
