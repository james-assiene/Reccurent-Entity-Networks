#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 21:16:14 2019

@author: assiene
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class OutputModule(nn.Module):
    
    def __init__(self, embedding_dim, output_dim):
        
        super(OutputModule, self).__init__()
        self.H = nn.Linear(embedding_dim, embedding_dim)
        self.R = nn.Linear(embedding_dim, output_dim)
        
        self.non_linearity = nn.PReLU()
        
    def forward(self, q, h):
        
        #q : batch x embedding_dim x 1
        #h : batch x num_memory_blocks x embedding_dim
        
        p = F.softmax(h.bmm(q), dim=1) # batch x num_memory_block x 1
        u = (p * h).sum(1, keepdim=True).transpose(1,2) # batch x embedding_dim x 1
        y = self.non_linearity(q.squeeze(2) + self.H(u.squeeze(2))) # batch x embedding_dim
        y = self.R(y) # batch x output_dim
        
        return y