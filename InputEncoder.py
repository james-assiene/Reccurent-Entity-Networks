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
        self.max_sequence_length = sequence_length
        self.f = torch.empty(self.max_sequence_length, embedding_dim)
        nn.init.xavier_normal_(self.f)
        
    def forward(self, input_sequence):
        
        e = self.input_embedding(input_sequence) # batch x num_sentences x num_words x embedding_dim
        
        s = self.f[:e.shape[2],:] * e
        s = s.sum(dim=2) # batch x num_sentences x embedding_dim
        
        return s