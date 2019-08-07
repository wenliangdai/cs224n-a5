#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
from torch import nn
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, word_embed_size: int):
        # word_embed_size is the dimension of word embedding produced by ConvNet
        super(Highway, self).__init__()

        self.linear_proj = nn.Linear(word_embed_size, word_embed_size)
        self.linear_gate = nn.Linear(word_embed_size, word_embed_size)

    def forward(self, x: torch.Tensor):
        # x is of size (batch_size, word_embed_size)
        x_proj = F.relu(self.linear_proj(x))
        x_gate = torch.sigmoid(self.linear_gate(x))
        # print(x_proj, x_gate)
        x_highway = x_proj * x_gate + (1 - x_gate) * x
        return x_highway

### END YOUR CODE
