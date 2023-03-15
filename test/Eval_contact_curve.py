import os
import sys
sys.path.append("..") 
import torch 
import torch.nn as nn
from inputs.inputs import *
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc,precision_recall_curve
from argparse import ArgumentParser


parser = ArgumentParser(description='')
parser.add_argument('-weight', default="/data/home/huangjiajian/code/Github/distance_predicted/weights/25.pth")
parser.add_argument('-feature', default="/assets_paper/huangjiajian/baker_attention_2/")
parser.add_argument('-label', default="/data/home/huangjiajian/code/Github/distance_predicted/datasets/Baker/distance/")
args = parser.parse_args()
model = torch.load(args.weight)


testdata = TestData(get_test_data,args.feature,args.label)
testdata_loader = DataLoader(testdata, batch_size=1, shuffle=False, drop_last=True)



AUC_ROC_all = 0 
AUC_PR_all = 0 
num = 0



for inputs,label,L in testdata_loader:
    outputs = model(inputs,L)
    outputs = outputs.squeeze()
    outputs = 1-outputs
    outputs = (outputs/0.92)*0.5
    outputs = outputs.to(torch.float64)

    label = label.squeeze()
    label = torch.floor(label).to(int)
    label = torch.where(label == 0 , 49, label)
    label = torch.where(label <= 8 , 1, 0).to(torch.float64)

    outputs = outputs.reshape(outputs.size()[-1]*outputs.size()[-2])
    label = label.reshape(label.size()[-1]*label.size()[-2])



    y_test = label.detach().cpu().numpy()
    y_score = outputs.detach().cpu().numpy()
    fpr,tpr,thre = roc_curve(y_test,y_score)
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)

    AUC_ROC = auc(fpr,tpr)
    AUC_ROC_all = AUC_ROC_all + AUC_ROC

    AUC_PR = auc(recall, precision)
    AUC_PR_all = AUC_PR_all + AUC_PR


    num = num + 1



print('ROC',AUC_ROC_all/num)
print('PR',AUC_PR_all/num)