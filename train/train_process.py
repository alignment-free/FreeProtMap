import numpy as np
import torch
from tensorboardX import SummaryWriter



def train(model,criterion,optimizer,args,traindata_loader):
    softmax = torch.nn.Softmax(dim=1)
    for epoch_num in range(args.epoch):
        for inputs,label,L in traindata_loader:

            inputs = inputs.squeeze()

            outputs = model(inputs,L)
  
            value = torch.floor(label).to(torch.int)
            value = torch.where(value == 0 , 0, 1)
            value = value.unsqueeze(1)
            outputs = value*outputs

            value = torch.where(label < 36 , 1, 0)
            label = label * value
            value = value.unsqueeze(1)
            outputs = outputs * value
            

            label = label/100
            label = label.unsqueeze(1)

            loss = criterion(outputs,label)       
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()





