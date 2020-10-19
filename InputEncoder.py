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
    """ The encoding layer summarizes an element of the input sequence with a vector of fixed length """
    
    def __init__(self, vocabulary_size, embedding_dim, max_sequence_length):
        
        super(InputEncoder, self).__init__()
        self.input_embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.max_sequence_length = max_sequence_length
        self.f = nn.Parameter(torch.empty(self.max_sequence_length, embedding_dim))
        nn.init.xavier_normal_(self.f)
        
    def forward(self, input_sequence):
        """Compute a fixed length representation of the input sequence of word embeddings

        Args:
            input_sequence (tensor): Input sequence of word embeddings

        Returns:
            tensor: Fixed length vector representing the input sequence
        """
        
        # Converts the "list" of tokens indices into a tensor of word embeddings
        e = self.input_embedding(input_sequence) # batch_size x num_sentences x num_words x embedding_dim
        
        # self.f[:e.shape[2],:] is (num_words x embedding_dim). We use broadcasting
        s = self.f[:e.shape[2],:] * e
        s = s.sum(dim=2) # batch_size x num_sentences x embedding_dim
        
        return s