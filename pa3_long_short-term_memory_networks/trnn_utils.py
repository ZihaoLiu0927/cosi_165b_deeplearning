import numpy as np
import pickle as pkl
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import *

torch.manual_seed(0)
c_len = 100
embed_d = 128

# data class
class input_data():
    def load_text_data(self, word_n = 100000):
        f = open('../dataset/paper_abstract.pkl', 'rb')
        p_content_set = pkl.load(f)
        f.close()

        p_label = [0] * 21044
        label_f = open('../dataset/paper_label.txt', 'r')
        for line in label_f:
            line = line.strip()
            label_s = re.split('\t',line)
            p_label[int(label_s[0])] = int(label_s[1])
        label_f.close()

        def remove_unk(x):
            return [[1 if w >= word_n else w for w in sen] for sen in x]

        p_content, p_content_id = p_content_set
        p_content = remove_unk(p_content)

        # padding with max len 
        for i in range(len(p_content)):
            if len(p_content[i]) > c_len:
                p_content[i] = p_content[i][:c_len]
            else:
                pad_len = c_len - len(p_content[i])
                p_content[i] = np.lib.pad(p_content[i], (0, pad_len), 'constant', constant_values=(0,0))

        p_id_train = []
        p_content_train = []
        p_label_train = []
        p_id_test = []
        p_content_test = []
        p_label_test = []
        for j in range(len(p_content)):
            if j % 10 in (3, 6, 9):
                p_id_test.append(p_content_id[j])
                #p_content_test.append(p_content[j])
                p_label_test.append(p_label[j])
            else:
                p_id_train.append(p_content_id[j])
                #p_content_train.append(p_content[j])
                p_label_train.append(p_label[j])

        p_train_set = (p_id_train, p_label_train)
        p_test_set = (p_id_test, p_label_test)

        return p_content, p_train_set, p_test_set


    def load_word_embed(self):
        #return word_embed
        word_embed = np.zeros((32784, 128))
        index = 0
        with open("../dataset/word_embed.txt", 'r') as f:
            for line in f:
                line1 = line.split()
                if (len(line1) != 129):
                    continue
                word_embed[index] = np.array(line1[1:])
                index += 1
        return word_embed


#text RNN Encoder
class Text_Encoder(nn.Module):
    def __init__(self, p_content, word_embed):
        # two input: p_content - abstract data of all papers, word_embed - pre-trained word embedding 
        super(Text_Encoder, self).__init__()
        self.X = p_content
        self.embed = word_embed
        # specify padding??
        
        self.lstmcell = nn.LSTMCell(128, 64, dtype=torch.float64)
        self.layer1 = nn.Linear(64, 32, dtype=torch.float64)
        self.layer2 = nn.Linear(32, 5, dtype=torch.float64)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, id_batch):
        # id_batch: use id_batch (paper ids in this batch) to obtain paper conent of this batch
        x_batch = []
        for id in id_batch:
            row = self.X[id.int()]
            p = np.zeros((100, 128))
            count = 0
            for w in row:
                p[count] = self.embed[w]
                count+=1
            x_batch.append(torch.from_numpy(p).to("cuda:0"))

        x_batch = torch.stack(x_batch)
        x_batch = x_batch.to(torch.float64).to("cuda:0")
             
        #outputs = []
        output = torch.zeros(x_batch.shape[0], 64, dtype=torch.float64).to("cuda:0")
        hx = torch.randn(x_batch.shape[0], 64, dtype=torch.float64).to("cuda:0")
        cx = torch.randn(x_batch.shape[0], 64, dtype=torch.float64).to("cuda:0")
        for i in range(x_batch.shape[1]):
            hx, cx = self.lstmcell(x_batch[:,i,:], (hx, cx))
            output = output + hx
            #outputs.append(hx)
        
        output = output / x_batch.shape[0]
        x = self.layer1(output)
        #x = self.layer1(outputs[-1])
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x
    
    
    
        






