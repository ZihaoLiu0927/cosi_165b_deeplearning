import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle as pkl
import dnn_utils as U


# PB-I-5 write model function 
def model(x_train, y_train, x_test, y_test, layers_dims, learning_rate, num_iter = 3000, print_loss = True):
	np.random.seed(1)
	loss_all = []
	
	# initialize pars (1 line of code)
	pars = U.initialize_pars_deep(layers_dims)
	
	for i in range(num_iter):
		# forward propagation (1 line of code)
		xL, caches = U.L_model_forward(x_train, pars)

		# compute loss (1 line of code)
		loss = U.compute_loss(xL, y_train)
		
		# backward propagation (1 line of code)
		grads = U.L_model_backward(xL, y_train, caches)
 		
 		# pars update (1 line of code)
		pars = U.update_pars(pars, grads, learning_rate)
				
		# record loss at different epoches (use loss_all)
		if print_loss:
			print(f'Loss after iteration {i+1} is: {loss}')
		loss_all.append(loss)

	# predict test/train data samples (2 lines of code)
	y_predict_test = U.predict(pars, x_test)
	y_predict_train = U.predict(pars, x_train)

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
		 "pars" : pars}
	return result


if __name__ == '__main__':
	# load datasets (1 line of code)
	train_data_x, train_data_y, test_data_x, test_data_y = U.load_dataset()

	# PA-I-4 reshape train_data_x/test_data_x as 2D arrays (2 lines of code)
	train_data_x = np.reshape(train_data_x, (train_data_x.shape[0], train_data_x.shape[1] * train_data_x.shape[2] * train_data_x.shape[3])).T
	test_data_x =  np.reshape(test_data_x, (test_data_x.shape[0], test_data_x.shape[1] * test_data_x.shape[2] * test_data_x.shape[3])).T

	# normalize data 
	train_data_x = train_data_x / 255.0
	test_data_x = test_data_x / 255.0
 	
	# run model (1 line of code), layers_dims = [12288, 32, 16, 1]
	layers_dims = [12288, 32, 16, 1]
	model(train_data_x, train_data_y, test_data_x, test_data_y, layers_dims, 0.01, 5000, True)





