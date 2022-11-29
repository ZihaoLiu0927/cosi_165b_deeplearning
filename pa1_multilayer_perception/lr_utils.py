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
    y = 1 / (1 + np.exp(- x))
    
    return y


# PA-II-1 write initialize_zero function for weight parameters w of dimension (dim, 1) and bias scalar b
# initialize pars with all zeros
def initialize_zero(dim):
    # input: dim -- size of w
    # returns: initialized w, b 
    w = np.zeros(dim)
    b = 0
    return w, b


# PA-II-2 write forward function for computing loss and gradient
def forward(w, b, x, y):
    # input:
    # w -- weight parameters
    # b -- bias parameter
    # x -- input data (data_size, number of data samples)
    # y -- label of data: 1 or 0

    a = sigmoid(np.dot(w, x.T) + b)
    m = len(x)
    dw = (1/m) * np.dot(a - y, x)
    db = (1/m) * np.sum(a - y)
    loss = -1*(1/m) * np.sum(
        y * np.log(a + 1e-9) + (1 - y) * np.log(1 - a + 1e-9)
    )
    # return:
    # dw -- gradient of the objective wrt w 
    # db -- gradient of the objective wrt b 
    # loss -- negative log-likelihood loss for logistic regression
    return dw, db, loss


# PA-II-3 write optimize function for parameter update
def optimize(w, b, x, y, num_iter, learning_rate, print_loss = True):
    # input: w, b -- parameters, x -- data, y -- label of data, num_iter: number of total iterations
    # output: w, b -- optimized parameters, loss_all -- recored loss at different epoches
    loss_all = []
    for i in range(num_iter):
        dw, db, loss = forward(w, b, x, y)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        loss_all.append(loss)
        if print_loss:
            print(f'The loss after the {i+1}-th iteration is: {loss}')
    return {"w": w, "b": b}, loss_all


# PA-II-4 write prediction function
def predict(w, b, x):
    # input: x -- input data, w, b -- learned pars
    # return: y_predict -- prediction values (labels) of data
    res = sigmoid(np.dot(w, x.T) + b)
    res[res<0.5] = 0
    res[res>=0.5] = 1
    return res



