import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

# data loader function, return four datasets: train_data_x, train_data_y, test_data_x, test_data_y
def load_dataset():
        # must convert dataset to tensor format in order to use TensorDataset module
        with open("C:\\Users\\Liuxi\\OneDrive\\桌面\\AS-2\\data\\test_data_x.pkl", 'rb') as f:
            test_data_x = pkl.load(f, encoding = 'latin1')
        with open("C:\\Users\\Liuxi\\OneDrive\\桌面\\AS-2\\data\\test_data_y.pkl", 'rb') as f:
            test_data_y = pkl.load(f, encoding = 'latin1')
            #test_data_y = torch.from_numpy(test_data_y).T
        with open("C:\\Users\\Liuxi\\OneDrive\\桌面\\AS-2\\data\\train_data_x.pkl", 'rb') as f:
            train_data_x = pkl.load(f, encoding = 'latin1')
        with open("C:\\Users\\Liuxi\\OneDrive\\桌面\\AS-2\\data\\train_data_y.pkl", 'rb') as f:
            train_data_y = pkl.load(f, encoding = 'latin1')
        return train_data_x, train_data_y, test_data_x, test_data_y

# CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            # Layer 1: input shape (64 * 64 * 3 figure); output shape (64 * 64 * 6); 64 - 5 + 1 + padding(2) * 2 = 64
            nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5, stride = 1, padding=2), 
            nn.ReLU(),
            # output shape (32 * 32 * 6)
            torch.nn.MaxPool2d(kernel_size=2),

            # Layer 2: input shape (32 * 32 * 6); output shape (32 * 32 * 12); 32 - 5 + 1 + padding(2) * 2 = 32
            nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 5, stride = 1, padding=2),
            torch.nn.ReLU(),
            # output shape (16 * 16 * 12)
            nn.MaxPool2d(kernel_size=2),
            
            # dropout
            torch.nn.Dropout(p = 0.2),
  
            nn.Flatten(),
            nn.Linear(16*16*12, 120),
            nn.ReLU(),
            nn.Linear(120, 64),
            nn.ReLU(),
            
            # dropout
            torch.nn.Dropout(p = 0.2),
            
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )
    
    # Progresses data across layers    
    def forward(self, x):
        return self.model(x)









