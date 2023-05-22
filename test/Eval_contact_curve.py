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
parser.add_argument('-weight', default="")
parser.add_argument('-feature', default="")
parser.add_argument('-label', default="")
args = parser.parse_args()
model = torch.load(args.weight)


testdata = TestData(get_test_data,args.feature,args.label)
testdata_loader = DataLoader(testdata, batch_size=1, shuffle=False, drop_last=True)

y_test_all = torch.from_numpy(np.array([]))
y_score_all = torch.from_numpy(np.array([]))


for inputs,label,L in testdata_loader:
    outputs,certain = model(inputs,L)
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

    y_test = np.concatenate((y_test_all, y_test), axis=0, out=None, dtype=None, casting="same_kind")
    y_score = np.concatenate((y_score_all, y_score), axis=0, out=None, dtype=None, casting="same_kind")




fpr,tpr,thre = roc_curve(y_test,y_score)
precision, recall, thresholds = precision_recall_curve(y_test, y_score)

AUC_ROC = auc(fpr,tpr)
AUC_PR = auc(recall, precision)






print('ROC',AUC_ROC)
print('PR',AUC_PR)
