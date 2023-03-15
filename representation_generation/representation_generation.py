import os
import sys
sys.path.append("..") 
from einops import rearrange

import torch 
import torch.nn as nn
from inputs.inputs import *
import numpy as np
import esm

group_dim = 36

model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter = alphabet.get_batch_converter()

source_dir = '/assets_paper/huangjiajian/baker_fasta/'
save_dir = '/assets_paper/huangjiajian/baker_attention/'



Files = os.listdir(source_dir)


model = model.cuda()
for File in Files:
    File = open(source_dir+File,'r')
    File = File.readlines()
    name = File[0][1:].strip()
    seq = File[1].strip()

    if (len(seq) > 700):
        continue

    L = len(seq)

    data = [(name, seq)]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        batch_tokens = batch_tokens.cuda()
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    outputs =  results["attentions"][:,:,:, 1: batch_lens-1,1: batch_lens-1]
    outputs =  outputs.squeeze()

    outputs = rearrange(outputs, 'l c h w -> (l c) h w')

    outputs =  rearrange(outputs, '(g c) h w -> g c h w', g =group_dim)

    pred_all = outputs.detach().cpu().numpy()
    pred = np.max(pred_all, axis=1)
    np.save(save_dir+name.strip(),pred)



