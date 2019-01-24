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
        
    def forward(self, q, h):
        p = F.softmax(q @ h.t(), dim=1)
        u = (p.t() * h).sum(0).view(1,-1)
        y = q + self.H(u)
        y = self.R(F.prelu(y))
        
        return y