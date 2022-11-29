import numpy as np
import matplotlib.pyplot as plt
# import scipy
import rcnn_utils as U
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
torch.manual_seed(0)

# model train
def model_train(train_data_x, train_data_y, test_data_x, test_data_y, optim_type = "adam", epochs_size = 100, printType = "loss"):
	#Selecting the appropriate training device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	net = U.Net().to(device)
			
	test_data_x = np.swapaxes(torch.Tensor(test_data_x), 1, 3)
	test_data_y = torch.Tensor(test_data_y.reshape(48))
	train_data_x = np.swapaxes(torch.Tensor(train_data_x), 1, 3)
	train_data_y = torch.Tensor(train_data_y.reshape(205))

	#Generating data loaders from the corresponding datasets
	batch_size = 5
	training_dataset = TensorDataset(train_data_x, train_data_y)
	train_loader = DataLoader(training_dataset, batch_size=batch_size)

	#Defining the model hyper parameters

	criterion = nn.CrossEntropyLoss()
	if optim_type == "adam":
		optimizer = optim.Adam(net.parameters(), lr=0.0005)
	elif optim_type == "SGD":
		optimizer = optim.SGD(net.parameters(), lr=0.01)
	elif optim_type == "Adagrad":
		optimizer = optim.Adagrad(net.parameters(), lr=0.005, weight_decay=0.01)
	else:
		return
	print("Start optimization, method: " + optim_type)
	#Training process begins
	train_loss_list = []
	test_accuracy_list = []
	for epoch in range(epochs_size):
		running_loss = 0

		for _, data in enumerate(train_loader):
			#Extract data point
			images = data[0].to(device)
			labels = data[1].type(torch.LongTensor).to(device)
	
			#Calculating loss
			outputs = net(images)
			loss = criterion(outputs, labels)
	
			#Clean the previous batch grad info
			optimizer.zero_grad()
			#Updating weights according to calculated loss
			loss.backward()
			optimizer.step()
			#Calculate running loss cumulation
			running_loss += loss.item()
        
		#Store the loss info
		train_loss_list.append(running_loss/len(train_loader))
		#Store test accuracy
		running_accuracy = model_test(test_data_x, test_data_y, net, epochs_size)
		test_accuracy_list.append(running_accuracy)
		if printType == "loss":
			#Printing loss for each epoch
			print(f"Training loss at iteration {epoch+1} = {train_loss_list[-1]}")
		elif printType == "accurary":
			#Printing test accuracy for each epoch
			print(f"Testing accuracy at iteration {epoch+1} = {running_accuracy}")

	accuracy = model_test(test_data_x, test_data_y, net, epochs_size)
	print(f'Training finished. The final model accuracy on test dataset is: {accuracy}\n')
	return train_loss_list, test_accuracy_list

# model test: can be called directly in model_train
def model_test(test_data_x, test_data_y, net, epoch_num):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	test_dataset = TensorDataset(test_data_x, test_data_y)
	test_loader = DataLoader(test_dataset, batch_size=5)

	correct = 0
	total = 0
	for data in test_loader:
		images, labels = data
        
		images = images.to(device)
		labels = labels.to(device)
		outputs = net(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
	return 100 * correct / total


if __name__ == '__main__':
	# load datasets
	train_data_x, train_data_y, test_data_x, test_data_y = U.load_dataset()

	# rescale data 
	train_data_x = train_data_x / 255.0
	test_data_x = test_data_x / 255.0


	# model train (model test function can be called directly in model_train)
	methods = ["adam", "SGD", "Adagrad"]
	results = []
	for m in methods:
		result = model_train(train_data_x, train_data_y, test_data_x, test_data_y, optim_type = m, epochs_size = 100, printType = None)
		results.append(result[1])
	results = pd.DataFrame(results)
	plt.plot(results.T)
	plt.legend(["adam", "SGD", "Adagrad"])










