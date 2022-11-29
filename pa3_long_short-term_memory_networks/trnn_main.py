import numpy as np
# import matplotlib.pyplot as plt
# import scipy
import trnn_utils as U
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(0)

# model train
def model_train(all_data, word_embed, train_data_x, train_data_y, test_data_x, test_data_y):
    device = torch.device("cuda:0")
        
    net = U.Text_Encoder(all_data, word_embed).to(device)

    
    batch_size  = 20
    training_dataset = TensorDataset(train_data_x, train_data_y)
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    epochs_size = 100
    

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    criterion.to(device)
    
    
    
    train_loss_list = []
    accuracies = []
    
    for epoch in range(epochs_size):
        running_loss = 0
        
        for _, data in enumerate(train_loader):
            # Extract batch data point
            papers, labels = data
            
            labels = labels.type(torch.LongTensor).to(device)
            
            papers = papers.to(device)
            
            # Calculate loss

            outputs = net(papers)
            
            outputs.to("cpu")

            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()
            
            running_loss += loss.item()
            
        
        train_loss_list.append(running_loss/len(train_loader))
        running_accurary = model_test(test_data_x, test_data_y, net, epochs_size)
        accuracies.append(running_accurary)
        print(f"Trainning accurary at iteration {epoch+1} is:  {running_accurary}%")
        print(f"Trainning loss at iteration {epoch+1} is:  {train_loss_list[-1]}")
    
    accuracy = model_test(test_data_x, test_data_y, net, epochs_size)
    print(f'The model accuracy on test dataset is: {accuracy}\n')
    return train_loss_list, accuracies

# model test: can be called directly in model_train 
def model_test(test_data_x, test_data_y, net, epoch_num):
    device = torch.device("cpu")
    test_dataset = TensorDataset(test_data_x, test_data_y)
    test_loader = DataLoader(test_dataset, batch_size=20)

    correct = 0
    total = 0
    for data in test_loader:
        papers, labels = data
        
        papers = papers.to("cuda:0")
        labels = labels.to("cuda:0")
        outputs = net(papers)
        predicted = torch.argmax(outputs.to("cuda:0"), 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100 * correct / total


if __name__ == '__main__':
	# load datasets
	input_data = U.input_data()

	all_data, train_data, test_data = input_data.load_text_data()
	train_data_x = torch.from_numpy(np.array(train_data[0])) # map content by id
	train_data_y = torch.from_numpy(np.array(train_data[1]))
	test_data_x = torch.from_numpy(np.array(test_data[0])) # map content by id
	test_data_y = torch.from_numpy(np.array(test_data[1]))

	word_embed = input_data.load_word_embed()

	# model train (model test function can be called directly in model_train)
	train_loss_list, accuracies = model_train(all_data, word_embed, train_data_x, train_data_y, test_data_x, test_data_y)






