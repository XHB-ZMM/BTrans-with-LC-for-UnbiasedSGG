import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class LogitCompensation(nn.Module):

    def __init__(self, cls_num_list, soft_factor,tau,weight=None):
        super(LogitAdjust, self).__init__()
        self.soft_factor = soft_factor
        self.tau = tau
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = self.soft_factor * cls_num_list / cls_num_list.sum()
        self.m_list = self.tau * torch.log(cls_p_list)
        self.m_list = self.m_list.view(1, -1)
     
    def forward(self, ori_dist):
        cor_dist = ori_dist + self.m_list
        return cor_dist

