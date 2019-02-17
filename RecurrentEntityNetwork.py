#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 23:35:43 2019

@author: assiene
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from InputEncoder import InputEncoder
from DynamicMemory import DynamicMemory
from OutputModule import OutputModule

class RecurrentEntityNetwork(nn.Module):
    
    def __init__(self, vocabulary_size=177, embedding_dim=100, sequence_length=7, num_memory_blocks=20):
        
        super(RecurrentEntityNetwork, self).__init__()
        self.input_encoder = InputEncoder(vocabulary_size, embedding_dim, sequence_length)
        self.dynamic_memory = DynamicMemory(num_memory_blocks, embedding_dim)
        self.output_module = OutputModule(embedding_dim, vocabulary_size)
        
    def forward(self, q, text):
        
        s = self.input_encoder(text)
        q = self.input_encoder(q.unsqueeze(1))
        
        for t in range(s.shape[1]):
            h = self.dynamic_memory(s[:,t,:])
        
        y = self.output_module(q.transpose(1,2), h)
        
        return y