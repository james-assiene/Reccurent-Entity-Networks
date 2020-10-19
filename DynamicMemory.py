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
    """The dynamic memory is a gated recurrent network with a (partially) block structured weight tying
    scheme

    
    """
    
    def __init__(self, num_mem_blocks=20, hidden_size=100):
        """Construct the memory cell

        Args:
            num_mem_blocks (int, optional): Number of blocks of the memory cell. Defaults to 20.
            hidden_size (int, optional): Size of an individual memory cell. Defaults to 100.
        """
        
        super(DynamicMemory, self).__init__()
        
        self.num_mem_blocks = num_mem_blocks
        self.hidden_size = hidden_size
        
        self.U = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.non_linearity = nn.PReLU()

    def reset_memory(self):
        """Reinitialise the memory cells with random content
        """

        self.h = torch.empty(1, self.num_mem_blocks, self.hidden_size)
        self.w = torch.empty(1, self.num_mem_blocks, self.hidden_size)

        nn.init.xavier_normal_(self.h)
        nn.init.xavier_normal_(self.w)
        
    def forward(self, s_t):
        """Update the content of the memory cells

        Args:
            s_t (tensor): Fixed length representation of the input sentence 't' (batch_size x embedding_dim)
        """
        
        s_t = s_t.unsqueeze(1) # batch_size x 1 x (embedding_dim=hidden_size)
        self.h = self.h.to(s_t.device)
        self.w = self.w.to(s_t.device)
        
        # h @ s_t.T : (1 x num_memory_blocks x hidden_size) @ (batch_size x hidden_size x 1)
        # Compute a "similarity score" between s_t and each memory block
        g = torch.sigmoid((self.h + self.w) @ s_t.transpose(1,2)) # batch_size x num_memory_blocks x 1
        # Compute candidate memory cells
        h_candidate = self.non_linearity(self.U(self.h) + self.V(self.w) + self.W(s_t)) # batch_size x num_memory_blocks x embedding_dim
        self.h = self.h + g * h_candidate # batch_size x num_memory_blocks x embedding_dim
        self.h = self.h / self.h.norm(p=2, dim=2, keepdim=True)
        
        return self.h