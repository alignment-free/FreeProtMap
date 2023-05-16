import os
import sys
sys.path.append("..") 
import torch 
import torch.nn as nn
from inputs.inputs import *
import numpy as np
import torch
from argparse import ArgumentParser


parser = ArgumentParser(description='')
parser.add_argument('-weight', default="/data/home/huangjiajian/code/Github/distance_predicted/weights/25.pth")
parser.add_argument('-feature', default="/assets_paper/huangjiajian/Ecoli_attention_2/")
parser.add_argument('-label', default="/data/home/huangjiajian/code/Github/distance_predicted/datasets/E.coli/distance/")
args = parser.parse_args()
model = torch.load(args.weight)


testdata = TestData(get_test_data,args.feature,args.label)
testdata_loader = DataLoader(testdata, batch_size=1, shuffle=False, drop_last=True)




def get_precision(y_hat,y):
    true_positive = torch.sum(torch.sum(torch.logical_and(y_hat, y)))
    predicted_positive = torch.sum(torch.sum(y_hat))+1e-9
    return true_positive / predicted_positive


def get_recall(y_hat,y):
    true_positive = torch.sum(torch.sum(torch.logical_and(y_hat, y)))
    all_positive = torch.sum(torch.sum(y))+1e-9
    return true_positive / all_positive



F1_all = 0
predicted_all = 0
real_all = 0
precision_all = 0
recall_all = 0
num = 0

for inputs,label,L in testdata_loader:

    num = num + 1 

    outputs,certain = model(inputs,L)
    outputs = outputs.squeeze()
    outputs = (1-outputs)/0.92*0.5
    outputs = torch.where(outputs < 0.5 , 0, 1)

    label = label.squeeze()
    label = torch.floor(label).to(int)
    label = torch.where(label == 0 , 49, label)
    label = torch.where(label <= 8 , 1, 0)



    precision = get_precision(outputs,label)
    recall = get_recall(outputs,label)
    precision_all = precision_all + precision
    recall_all = recall_all + recall
    F1       =  1/precision +1/recall
    F1      =   2/F1
    F1_all = F1_all + F1









print('F1',F1_all.item()/num)
print('PRECISION',precision_all.item()/num)
print('RECALL',recall_all.item()/num)





