#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 20:20:25 2019

@author: assiene
"""

import numpy as np
import torch
import torch.nn as nn

class InputEncoder(nn.Module):
    
    def __init__(self, vocabulary_size, embedding_dim, sequence_length):
        
        super(InputEncoder, self).__init__()
        self.input_embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.sequence_length = sequence_length
        self.f = torch.empty(sequence_length, embedding_dim)
        nn.init.xavier_normal_(self.f)
        
    def forward(self, input_sequence):
        
        e = self.input_embedding(input_sequence)
        
        s = self.f * e
        s = s.sum(0)
        
        return s.view(1,-1)