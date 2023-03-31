import os
import sys
sys.path.append("..")
from argparse import ArgumentParser
import torch 
import torch.nn as nn
from inputs.inputs import *
import numpy as np
import torch


parser = ArgumentParser(description='')
parser.add_argument('-weight', default="/data/home/huangjiajian/code/Github/distance_predicted/weights/25.pth")
parser.add_argument('-feature', default="/assets_paper/huangjiajian/baker_attention_2/")
parser.add_argument('-label', default="/data/home/huangjiajian/code/Github/distance_predicted/datasets/Baker/distance/")
args = parser.parse_args()
model = torch.load(args.weight)


testdata = TestData(get_test_data,args.feature,args.label)
testdata_loader = DataLoader(testdata, batch_size=1, shuffle=False, drop_last=True)


small_num = 0
Sum_MAE = 0
Sum_RMSE = 0
Sum_RS = 0

sum_num = 0

for inputs,label,L in testdata_loader:

    if ( int(L.item()) > 750):
        continue


    outputs = model(inputs,L)

    value = torch.floor(label).to(torch.int)
    value_1 = torch.where(value == 0 , 0, 1)
    value   = value_1
    num = torch.sum(torch.sum(value))
    value = value.unsqueeze(1)
    outputs = value*outputs.cuda().unsqueeze(0).unsqueeze(1)
    label = value*label.cuda().unsqueeze(0).unsqueeze(1)
    outputs = outputs.squeeze()
    label = label.squeeze()



    outputs_MSE = outputs*100
    MSE = (outputs_MSE-label)**2
    MSE = torch.sum(torch.sum(MSE))/num
    RMSE =MSE.item()** 0.5 
    Sum_RMSE = Sum_RMSE + RMSE

    value_mean = torch.sum(torch.sum(outputs_MSE))/num
    value_predict = outputs_MSE - label
    value_predict = torch.sum(torch.sum(value_predict**2))
    value_total = label - value_mean
    value_total = torch.sum(torch.sum(value_total**2))
    RS = 1- value_predict/ value_total
    Sum_RS = Sum_RS + RS.item()

    outputs = outputs*100
    D_value = outputs - label
    D_value = (torch.sum(torch.sum(torch.abs(D_value)))/num).item()
    Sum_MAE = Sum_MAE + D_value



    sum_num = sum_num + 1

MAE  = Sum_MAE / sum_num
RMSE = Sum_RMSE / sum_num
RS   = Sum_RS / sum_num



print('MAE',MAE)
print('RMSE',RMSE)
print('RS',RS)



