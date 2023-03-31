from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn.functional as F
import os


def get_train_data(dir_attention,distance_dir,files,item):

    name = files[item]

    inputs_attention = np.load(dir_attention+name)
    inputs = torch.from_numpy(inputs_attention) 

    label = np.load(distance_dir+ name.split('.')[0]+'.npz')
    label = label['dist_ca']

    L     =  inputs.shape[2]
    L     =  torch.ones(1)*L

    label = torch.from_numpy(label).squeeze(0)

    return inputs.cuda(),label.cuda(),L.cuda()

def get_test_data(dir_label,dir_feature,files,item):
    name = files[item]

    inputs = np.load(dir_feature + name)
    inputs = torch.from_numpy(inputs) 

    name = name.split('.')[0]+'.txt'
    label = np.loadtxt(dir_label + name)
    # label = np.loadtxt(dir_label +'label/'+ name)
    L     =  inputs.shape[2]
    L     =  torch.ones(1)*L
    label = torch.from_numpy(label).squeeze(0)

    return inputs.cuda(),label.cuda(),L.cuda()



def collate_fn(batch):
    pad_index = 0
    lens = [inputs.size(2) for inputs,label,L in batch]
    lens.sort()
    seq_len = lens[-1]

    pad_inputs_list = []
    pad_label_list = []
    pad_L_list = []
    batch_inputs_list = []
    batch_label_list = []
    batch_L_list = []

    for inputs,label,L in batch:

        pad_len = seq_len - inputs.size(2)

        label = F.pad(label, (0,pad_len,0,pad_len), "constant", 0)
        inputs = F.pad(inputs, (0,pad_len,0,pad_len), "constant", 0)

        pad_inputs_list.append(inputs)
        pad_label_list.append(label)
        pad_L_list.append(L)


    
    pad_inputs = torch.stack(pad_inputs_list, 0)
    pad_label = torch.stack(pad_label_list, 0)
    pad_L = torch.stack(pad_L_list, 0)

    return pad_inputs.cuda(),pad_label.cuda(),pad_L.cuda()



class MyData(Dataset):
    def __init__(self,train_num,feature,label,loader,transform=None, target_transform=None):
        super(MyData,self).__init__()
        self.feature =  feature
        self.label =   label


        self.transform=transform
        self.target_transform=target_transform
        self.loader=loader
        self.train_num = train_num
        self.files = os.listdir(self.feature)

    def __getitem__(self, item):
        inputs,label,L = self.loader(self.feature,self.label,self.files,item)
        return inputs,label,L


    def __len__(self):
        return self.train_num



class TestData(Dataset):
    def __init__(self,loader, test_feature,test_label,transform=None, target_transform=None):
        super(TestData,self).__init__()

        self.test_feature = test_feature
        self.test_label = test_label


        self.transform=transform
        self.target_transform=target_transform
        self.loader=loader
        self.files = os.listdir(self.test_feature)

    def __getitem__(self, item):
        inputs,label,L = self.loader(self.test_label,self.test_feature,self.files,item)
        return inputs,label,L


    def __len__(self):
        return len(self.files)
