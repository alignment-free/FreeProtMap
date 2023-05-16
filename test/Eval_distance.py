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
parser.add_argument('-weight', default="")


parser.add_argument('-feature', default="")
parser.add_argument('-label', default="")



args = parser.parse_args()
model = torch.load(args.weight)


testdata = TestData(get_test_data,args.feature,args.label)
testdata_loader = DataLoader(testdata, batch_size=1, shuffle=False, drop_last=True)


small_num = 0
Sum_MAE = 0
Sum_RMSE = 0
Sum_RS = 0
Sum_UNCERTAIN = 0

sum_num = 0


list_RMSE = torch.ones(90)
list_RS = torch.ones(90)
list_MAE = torch.ones(90)
list_uncertain = torch.ones(90)





with torch.no_grad():
    for inputs,label,L in testdata_loader:

        
        if ( int(L.item()) > 750):
            continue


        outputs,uncertain = model(inputs,L)


        value = torch.where(label < 36 , 1, 0)
        value = value.unsqueeze(1)



        outputs = outputs * value
        label = label * value
        uncertain = uncertain * value


        label = label.unsqueeze(1)





        ones = torch.ones(outputs.size()).cuda()
        eyes = torch.eye(outputs.size()[-1]).cuda()
        tmp = ones - eyes
        outputs = tmp*outputs
        distance = outputs * 100
        uncertain = uncertain * 100

        D_value = distance - label
        uncertain = uncertain - D_value
        uncertain = (torch.mean(torch.mean(torch.abs(uncertain)))).item()

        Sum_UNCERTAIN = Sum_UNCERTAIN + uncertain
        list_uncertain[sum_num] = uncertain




        D_value = (torch.mean(torch.mean(torch.abs(D_value)))).item()
        Sum_MAE = Sum_MAE + D_value
        list_MAE[sum_num] = D_value




        MSE = (distance-label)**2
        MSE = torch.mean(torch.mean(MSE))
        RMSE =MSE.item()** 0.5 
        Sum_RMSE = Sum_RMSE + RMSE
        list_RMSE[sum_num] = RMSE

        value_mean = torch.mean(torch.mean(distance))
        value_predict = distance - label
        value_predict = torch.mean(torch.mean(value_predict**2))
        value_total = label - value_mean
        value_total = torch.mean(torch.mean(value_total**2))
        RS = 1- value_predict/ value_total
        Sum_RS = Sum_RS + RS.item()
        list_RS[sum_num] = RS

        sum_num = sum_num + 1


MAE  = Sum_MAE / sum_num
RMSE = Sum_RMSE / sum_num
RS   = Sum_RS / sum_num
UNCERTAIN   = Sum_UNCERTAIN / sum_num



S_MAE = torch.abs((list_MAE-MAE))
S_MAE = torch.mean(S_MAE)

S_RMSE = torch.abs((list_RMSE-RMSE))
S_RMSE = torch.mean(S_RMSE)

S_RS = torch.abs((list_RS-RS))
S_RS = torch.mean(S_RS)


print('MAE',MAE)
print('RMSE',RMSE)
print('RS',RS)
print('UNCERTAIN',UNCERTAIN)

print('S_MAE',S_MAE)
print('S_RMSE',S_RMSE)
print('S_RS',S_RS)
