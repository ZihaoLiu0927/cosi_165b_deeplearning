import numpy as np
# import matplotlib.pyplot as plt
# import scipy
import gnn_utils as U
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# model train
def model_train(adj, features, labels, idx_train, idx_test, opt = "Adam", epoch = 300):
	model = U.GCN(nfeat=features.shape[1], nhid = 32, nclass = 7, dropout = 0.5)
	if opt == 'Adam':
		optimizer = optim.Adam(model.parameters(), lr = 0.005)
	elif opt == 'SGD':
		optimizer = optim.SGD(model.parameters(), lr = 0.01)
	elif opt == 'Adagrad':
		optimizer = optim.Adagrad(model.parameters())
	accuracy = []
	loss = []
	accuracy_test = []
	loss_test = []
	for _ in range(epoch):
		model.train()
		optimizer.zero_grad()
		output = model(features, adj)
		loss_train = F.nll_loss(output[idx_train], labels[idx_train])
		loss.append(loss_train.item())
		running_accuracy = U.accuracy(output[idx_train], labels[idx_train])
		accuracy.append(running_accuracy.item())
		loss_train.backward()
		optimizer.step()
		running_loss_test, running_accuracy_test = model_test(model, adj, features, labels, idx_test)
		accuracy_test.append(running_accuracy_test.item())
		loss_test.append(running_loss_test.item())
	return {"loss_train": loss, "acc_train": accuracy, "loss_test": loss_test, "acc_test": accuracy_test}

# model test: can be called directly in model_train 
def model_test(model, adj, features, labels, idx_test):
	output = model(features, adj)
	return F.nll_loss(output[idx_test], labels[idx_test]), U.accuracy(output[idx_test], labels[idx_test])

if __name__ == '__main__':
	# load datasets
	adj, features, labels, idx_train, idx_test = U.load_data()

	result = model_train(adj, features, labels, idx_train, idx_test)
	print(result['acc_test'][-1])
 
	result_adm = model_train(adj, features, labels, idx_train, idx_test)
	result_sgd = model_train(adj, features, labels, idx_train, idx_test, opt="SGD")
	result_ada = model_train(adj, features, labels, idx_train, idx_test, opt="Adagrad")
 
	acc_adam = result_adm["acc_test"]
	acc_sgd = result_sgd["acc_test"]
	acc_ada = result_ada["acc_test"]
	x = np.arange(300)
	plt.plot(x, acc_adam, 'b', label='Adam')
	plt.plot(x, acc_sgd, 'r', label='SGD')
	plt.plot(x, acc_ada, 'g', label='Adagrad')
	
	plt.title("Accuracy of 3 Optimizers")
	plt.legend()
	plt.show()
 
	# model train (model test function can be called directly in model_train)
	model_train(adj, features, labels, idx_train, idx_test)






