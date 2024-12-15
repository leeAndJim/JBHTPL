import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import normal
from scipy.special import ndtri


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)   #目标类概率
    loss = (1 - p.detach()) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=False, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 /np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)                             #one-hot
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))  #取得对应位置的m   self.m_list
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)                                       #x的index位置换成x_m
        
        return F.cross_entropy(self.s*output, target, weight=self.weight)  #weight=self.weight
    
 
class KPSLoss(nn.Module):
    r"""Implement of KPS Loss :
    Args:
    """

    def __init__(self, cls_num_list, max_m=0.5, weighted=False, weight= None, s=30):
        super(KPSLoss, self).__init__()
        assert s > 0

        s_list = torch.cuda.FloatTensor(cls_num_list)
        s_list = s_list*(6/s_list.min())
        s_list = torch.log(s_list) #torch.log(s_list) #s_list**(1/4) #torch.log(s_list) #s_list**(1/4)#s_list = torch.log(s_list)**2  #s_list**(1/5)
        s_list = s_list*(1/s_list.min()) #s+ s_list #
        self.s_list = s_list
        self.s = s
        
        m_list =  torch.flip(self.s_list, dims=[0])
        m_list = m_list * (max_m / m_list.max())
        self.m_list = m_list
                
        self.weighted = weighted
        self.weight = weight
        

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = input*self.s_list

        phi = cosine - self.m_list
        # --------------------------- convert label to one-hot ---------------------------
        index = torch.zeros_like(input, dtype=torch.uint8)
        index.scatter_(1, label.data.view(-1, 1), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        #output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output = torch.where(index, phi, cosine)

        if self.weighted == False:
            output *= self.s
        elif self.weighted == True:
            index_float = index.type(torch.cuda.FloatTensor)
            batch_s = torch.flip(self.s_list, dims=[0])*self.s
            batch_s = torch.clamp(batch_s, self.s, 50)    #s过大不好。          
            batch_s = torch.matmul(batch_s[None, :], index_float.transpose(0,1)) 
            batch_s = batch_s.view((-1, 1))           
            output *= batch_s
        else:
            output *= self.s
        return F.cross_entropy(output, label) #, weight= self.weight)



class IIFLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self,dataset,variant='raw',iif_norm=0,reduction='mean',device='cuda',weight=None):
        super(IIFLoss, self).__init__()
        self.loss_fcn = nn.CrossEntropyLoss(reduction='none',weight=weight)
        self.reduction=reduction
        self.variant = variant
        
        freqs = np.array(dataset)
        iif={}
        iif['raw']= np.log(freqs.sum()/freqs)
        iif['smooth'] = np.log((freqs.sum()+1)/(freqs+1))+1
        iif['rel'] = np.log((freqs.sum()-freqs)/freqs)
        
        iif['normit'] = -ndtri(freqs/freqs.sum())
        iif['gombit'] = -np.log(-np.log(1-(freqs/freqs.sum())))
        iif['base2'] = np.log2(freqs.sum()/freqs)
        iif['base10'] = np.log10(freqs.sum()/freqs)
        self.iif = {k: torch.tensor([v],dtype=torch.float).to(device,non_blocking=True) for k, v in iif.items()}
        if iif_norm >0:
            self.iif = {k: v/torch.norm(v,p=iif_norm)  for k, v in self.iif.items()}
        
    def forward(self, pred, targets=None,infer=False):
        if infer is False:
            loss = self.loss_fcn(pred*self.iif[self.variant],targets)
            # print(loss)
            if self.reduction=='mean':
                loss=loss.mean()
            elif self.reduction=='sum':
                loss=loss.sum()
            return loss
        else:
            out = (pred*self.iif[self.variant])
            return out


class CE_ABC_Loss(nn.Module):
    def __init__(self, cls_num_list):
        super(CE_ABC_Loss, self).__init__()
        t = 0.5
        self.A = torch.cuda.FloatTensor(cls_num_list)
        self.A = self.A * (1 / self.A.mean())
        self.A = torch.pow(self.A, t)

    def forward(self, input, label):
        bs, c_l = input.size()
        lambda_1 = 0.5
        input_soft = F.softmax(input, dim=1)
        p_hat = torch.pow(input_soft, 2)
        ABC_loss = p_hat * self.A
        ABC_loss = ABC_loss.sum() / bs
        CE_loss = F.cross_entropy(input, label)
        loss = lambda_1 * CE_loss + (1-lambda_1) * ABC_loss
        return loss
    
class KPS_ABC_Loss(nn.Module):
    def __init__(self, cls_num_list):
        super(KPS_ABC_Loss, self).__init__()
        self.loss_kps = KPSLoss(cls_num_list=cls_num_list, max_m=0.1, s=1, weighted=False)
        t = 0.5
        self.A = torch.cuda.FloatTensor(cls_num_list)
        self.A = self.A * (1 / self.A.mean())
        self.A = torch.pow(self.A, t)

    def forward(self, input, label):
        bs, c_l = input.size()
        lambda_1 = 0.9
        input_soft = F.softmax(input, dim=1)
        p_hat = torch.pow(input_soft, 2)
        ABC_loss = p_hat * self.A
        ABC_loss = ABC_loss.sum() / bs
        # CE_loss = F.cross_entropy(input, label)
        KPS_loss = self.loss_kps(input, label)
        loss = lambda_1 * KPS_loss + (1-lambda_1) * ABC_loss
        return loss