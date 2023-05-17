import os
import sys    
import torch 
import torch.nn as nn
torch.cuda.device_count() 
sys.path.append("..")
from argparse import ArgumentParser
from inputs.inputs import *
from train.train_process import train 
from model.R_former.R_former_model import R_former


parser = ArgumentParser(description='')
parser.add_argument('--log', default="")
parser.add_argument('-train_num', default=11000)      
parser.add_argument('-train_batchsize', default=8)
parser.add_argument('--feature', default='')
parser.add_argument('--label', default='')
parser.add_argument('--save_dir', default='')
parser.add_argument('-epoch', default=30)


if __name__ =="__main__":
        args = parser.parse_args()
   
        model = R_former(36).cuda()
        model = nn.DataParallel(model)

        criterion_L1 = nn.L1Loss(reduction='elementwise_mean')


        optimizer=torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)

        traindata = MyData(args.train_num,args.feature, args.label,get_train_data)
        traindata_loader = DataLoader(traindata, batch_size=args.train_batchsize, shuffle=False, drop_last=True, collate_fn=collate_fn)

        train(model,criterion_L1,optimizer,args,traindata_loader)
