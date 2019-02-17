#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 20:37:20 2019

@author: assiene
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicMemory(nn.Module):
    
    def __init__(self, hidden_number=20, hidden_size=100):
        
        super(DynamicMemory, self).__init__()
        self.h = torch.empty(1, hidden_number, hidden_size)
        self.w = torch.empty(1, hidden_number, hidden_size)
        
        self.U = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.non_linearity = nn.PReLU()
        
        nn.init.xavier_normal_(self.h)
        nn.init.xavier_normal_(self.w)
        
    def forward(self, s_t):
        
        # s_t : batch x embedding_dim
        s_t = s_t.unsqueeze(1)
        #print("s_t : ", s_t.shape, "\n h : ", self.h.shape)
        
        g = torch.sigmoid((self.h + self.w) @ s_t.transpose(1,2)) # batch x num_memory_blocks x 1
        h_candidate = self.non_linearity(self.U(self.h) + self.V(self.w) + self.W(s_t)) # batch x num_memory_blocks x embedding_dim
        self.h = self.h + g * h_candidate # batch x num_memory_blocks x embedding_dim
        self.h = self.h / self.h.norm(p=2, dim=2, keepdim=True)
        
        return self.h