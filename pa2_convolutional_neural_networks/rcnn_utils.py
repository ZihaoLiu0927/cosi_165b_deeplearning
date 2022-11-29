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

# RCNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
    
            # Layer 1: input shape (64 * 64 * 3 figure); output shape (64 * 64 * 6); 64 - 5 + 1 + padding(2) * 2 = 64
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5, stride = 1, padding=2)
        # output shape (32 * 32 * 6)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.block = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size = 1, stride = 1),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size = 1, stride = 1),
        )
        

        # Layer 4: input shape (32 * 32 * 6); output shape (32 * 32 * 12); 32 - 5 + 1 + padding(2) * 2 = 32
        self.conv4 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 5, stride = 1, padding=2)
        # output shape (16 * 16 * 12)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
  
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(16*16*12, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 2)
        
    
    # Progresses data across layers    
    def forward(self, x):
        # first layer
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # residual block as layer 2 and 3
        shortcut = x
        x = self.block(x)
        x = F.relu(x + shortcut)
        
        # 4th layer
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        # full connection layer
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x







