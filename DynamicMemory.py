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
        self.h = torch.empty(hidden_number, hidden_size)
        self.w = torch.empty(hidden_number, hidden_size)
        
        self.U = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, s):
        
        g = F.sigmoid(s @ (self.h.t() + self.w.t()))
        h_candidate = F.prelu(self.U(self.h) + self.V(self.w) + self.W(s))
        self.h = self.h + g.t() * h_candidate
        self.h = self.h / self.h.norm(dim=1).view(-1,1)
        
        return self.h