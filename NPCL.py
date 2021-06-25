import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def HardHingeLoss(logit,groundTruth):    
    Nc = logit.data.size()
    y_onehot = torch.FloatTensor(len(groundTruth), Nc[1]).cuda()
    
    
    y_onehot.zero_()
    y_onehot.scatter_(1, groundTruth.data.view(len(groundTruth),1), 1.0)     
    y = torch.autograd.Variable(y_onehot).cuda()
    t = logit*y
    L1 =torch.sum(t, dim=1)
    
    M,idx = logit.topk(2, 1, True, True)
    
    f1 = torch.eq(idx[:,0],groundTruth).float()
    u=  M[:,0]*(1-f1) + M[:,1]*f1


    L = torch.clamp(1.0-L1+u, min=0) 

    return L

def logsumexp(inputs, dim=None, keepdim=False):
    return (inputs - F.log_softmax(inputs,dim)).mean(dim, keepdim=keepdim)





def SoftHingeLoss(logit,groundTruth):
    Nc = logit.data.size()
    y_onehot = torch.FloatTensor(len(groundTruth), Nc[1]).cuda()
        
    y_onehot.zero_()
    y_onehot.scatter_(1, groundTruth.data.view(len(groundTruth),1), 1.0)
    
    y = torch.autograd.Variable(y_onehot).cuda()
    t = logit*y
    L1 =torch.sum(t, dim=1)
    M,idx = logit.topk(2, 1, True, True)

    f1 = torch.eq(idx[:,0],groundTruth).float()

    u = logsumexp(logit,dim=1)*(1-f1) + M[:,1]*f1

    L = torch.clamp(1.0-L1+u, min=0) 

    return L



def loss_NPCL(y_1,  t,  Lrate,Nratio = 0):
###    
#  y_1 : prediction logit 
#  t   : target
# Lrate:  true/false  at the initiliztion phase (first a few epochs) set false to train with an upperbound ;
#                     at the working phase , set true to traing with NPCL. 
# Nratio:  noise ratio , set to zero for the clean case(it becomes CL when setting to zero)

###
    loss_1 = HardHingeLoss(y_1,t)
    ind_1_sorted = np.argsort(loss_1.data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    epsilon = Nratio

    if Lrate :
   
        Ls = torch.cumsum(loss_1_sorted,dim=0)
        B =  torch.arange(start= 0 ,end=-len(loss_1_sorted),step=-1)
        B = torch.autograd.Variable(B).cuda()
        _, pred1 = torch.max(y_1.data, 1)
        E = (pred1 != t.data).sum()
        C = (1-epsilon)**2 *  float(len(loss_1_sorted)) + (1-epsilon) *  E
        B = C + B
        mask = (Ls <= B.float()).int()
        num_selected = int(sum(mask))
        Upbound = float( Ls.data[num_selected-1] <= ( C.float() - num_selected))   # footnate in the paper
        num_selected = int( min(  round(num_selected + Upbound), len(loss_1_sorted) ))
    
        ind_1_update = ind_1_sorted[:num_selected]
    
        loss_1_update = SoftHingeLoss( y_1[ind_1_update], t[ind_1_update])
        
    else:
        loss_1_update = SoftHingeLoss( y_1, t)
        


    return torch.mean(loss_1_update)
