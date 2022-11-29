import numpy as np
import pickle as pkl


# PA-I-1 write data loader function, return four datasets: train_data_x, train_data_y, test_data_x, test_data_y
def load_dataset():
    with open("/Users/zach/Downloads/cosi-165b/AS-1/datasets/test_data_x.pkl", 'rb') as f:
        test_data_x = pkl.load(f, encoding = 'latin1')
    with open("/Users/zach/Downloads/cosi-165b/AS-1/datasets/test_data_y.pkl", 'rb') as f:
        test_data_y = pkl.load(f, encoding = 'latin1')
    with open("/Users/zach/Downloads/cosi-165b/AS-1/datasets/train_data_x.pkl", 'rb') as f:
        train_data_x = pkl.load(f, encoding = 'latin1')
    with open("/Users/zach/Downloads/cosi-165b/AS-1/datasets/train_data_y.pkl", 'rb') as f:
        train_data_y = pkl.load(f, encoding = 'latin1')
    return train_data_x, train_data_y, test_data_x, test_data_y

def sigmoid(x):
	y = 1.0 / (1 + np.exp(-x))
	
	return y, x


def relu(x):
	y = np.maximum(0, x)
	
	return y, x


def relu_backward(dy, x):
	dx = np.array(dy, copy = True) 
	
	dx[x <= 0] = 0
	
	return dx


def sigmoid_backward(dy, x):
	s = 1.0 / (1 + np.exp(-x))
	dx = dy * s * (1 - s)
	
	return dx


# PB-I-1 write initialize_pars_deep function for weight parameters w and bias scalar b
# w are initialized with standard normal function * 0.1, b are initialized with 0 
def initialize_pars_deep(layer_dims):
	#input: layer_dims --  the dimensions of each layer in our network
	# returns: parameters -- w1, b1, ..., wL, bL:
	#          wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
	#          bl -- bias vector of shape (layer_dims[l], 1)
	np.random.seed(3) # fix seed
	pars = {}
	number_layers = len(layer_dims)
	for i in range(1, number_layers):
		pars['w' + str(i)] = np.random.randn(
			layer_dims[i],
			layer_dims[i-1]
		) * 0.1
		
		pars['b' + str(i)] = np.zeros((layer_dims[i], 1))
	return pars

# linear forward function
def linear_forward(x, w, b):
	y = np.dot(w, x) + b
	
	cache = (x, w, b)
	
	return y, cache


# PB-I-2 write linear_activation_forward function for computing activations of each layer
def linear_activation_forward(x, w, b, activation):
	# input:
	# x -- activations from previous layer (or input data)
	# w -- weights matrix, numpy array of shape (size of current layer, size of previous layer)
	# b -- bias vector, numpy array of shape (size of the current layer, 1)
	# activation: activation function: sigmoid or relu

	# return:
	# y -- the output of the activation function
	# cache -- tuple of (linear_cache, activation_cache), linear_cache and activation_cache: the second output of linear_forward and sigmoid/relu
	# cache is stored for computing the backward
	z, linear_cache = linear_forward(x, w, b)
	if activation == "relu":
		y, activation_cache = relu(z)
	elif activation == "sigmoid":
		y, activation_cache = sigmoid(z)
	return y, (linear_cache, activation_cache)



# L_model_forward function for computing forward of each layer
# hidden layer activation: relu, output layer activation: sigmoid
# use linear_activation_forward function
def L_model_forward(x, pars):
	# input: x -- input data, pars -- initialized pars
	# return: xL -- output values, caches -- stored values (the second output of linear_activation_forward) of all layers 
	caches = []
	L = len(pars) // 2
	y = x
	
	for l in range(1, L):
		x_prev = y
		y, cache = linear_activation_forward(x_prev, pars['w' + str(l)], pars['b' + str(l)], "relu")
		caches.append(cache)
	
	xL, cache = linear_activation_forward(y, pars['w' + str(L)], pars['b' + str(L)], "sigmoid")
	
	caches.append(cache)
			
	return xL, caches


# PB-I-3 write compute_loss function for computing loss
def compute_loss(y_hat, y):
	# input: y_hat -- prediction value, y -- groundtruth value of shape (1, number of data samples)
	# return: loss -- loss value
	L = y.shape[1]
	loss = -np.sum(
		np.multiply(np.log(y_hat), y) + np.multiply(1 - y, np.log(1 - y_hat))
	) / L
	return np.squeeze(loss)

def linear_backward(dy, cache):
	x_prev, w, b = cache
	num = x_prev.shape[1]

	dw = 1.0 / num * np.dot(dy, x_prev.T)
	db =  1.0 / num * (np.sum(dy, axis = 1, keepdims = True))
	dy_prev = np.dot(w.T, dy)
	
	return dy_prev, dw, db


# linear_activation_backward function for computing backward gradients
# use relu_backward/sigmoid_backward and linear_backward functions
def linear_activation_backward(dy, cache, activation):
	# input:
	# dy -- post-activation gradient for current layer l 
	# cache -- tuple of (linear_cache, activation_cache) for computing backward propagation 
	# activation -- the activation to be used in this layer: "sigmoid" or "relu"
	
	# return:
	# dy_prev -- gradient of the objective wrt the activation of the previous layer l-1
	# dw -- gradient of the objective wrt w of current layer l
	# db -- gradient of the objective wrt b of current layer l

	linear_cache, activation_cache = cache
	
	if activation == "relu":
		dx = relu_backward(dy, activation_cache)
		dy_prev, dw, db = linear_backward(dx, linear_cache)
		
	elif activation == "sigmoid":
		dx = sigmoid_backward(dy, activation_cache)
		dy_prev, dw, db = linear_backward(dx, linear_cache)
	
	return dy_prev, dw, db



# L_model_backward function for computing backward of each layer
# use linear_activation_backward function
def L_model_backward(xL, y, caches):
	# input:
	# xL -- prediction values
	# y -- groundtruth values of all data samples
	# caches -- activations of all layers: the second output of L_model_forward

	# return: grads -- gradients of the objective wrt activation and parameters (dy, dw, db) of all layers 
	grads = {}
	L = len(caches) 
	num = xL.shape[1]
	y = y.reshape(xL.shape) 
	
	# initializing the backpropagation
	dxL = - (np.divide(y, xL) - np.divide(1 - y, 1 - xL))

	current_cache = caches[L-1]
	grads["dy" + str(L)], grads["dw" + str(L)], grads["db" + str(L)] = linear_activation_backward(dxL, current_cache, "sigmoid")

	for l in reversed(range(L-1)):
		current_cache = caches[l]
		dy_prev_temp, dw_temp, db_temp = linear_activation_backward(grads["dy" + str(l + 2)], current_cache, "relu")
		grads["dy" + str(l + 1)] = dy_prev_temp
		grads["dw" + str(l + 1)] = dw_temp
		grads["db" + str(l + 1)] = db_temp

	return grads



def update_pars(pars, grads, learning_rate):	
	L = len(pars) // 2
	for i in range(1, L + 1):
		pars["w" + str(i)] = pars["w" + str(i)] - learning_rate * grads["dw"+str(i)]
		pars["b" + str(i)] = pars["b" + str(i)] - learning_rate * grads["db"+str(i)]
		
	return pars


# PB-I-4 write prediction function for model prediction
def predict(pars, x):
	# input: x -- input data, pars -- learned pars
	# return: y_predict -- prediction values of data samples 
	xL, _ = L_model_forward(x, pars)
	xL[xL < 0.5] = 0
	xL[xL >= 0.5] = 1
	return xL



