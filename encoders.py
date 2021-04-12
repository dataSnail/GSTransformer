# -*- coding: utf-8 -*-
'''
@Time    : 2021-04-12 9:25 p.m.
@Author  : datasnail
@File    : encoders.py.py
'''

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np

class GSTransformer(nn.Module):
    def __init__(self, input_dim):
        self.transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                                          dim_feedforward=2048, dropout=0.1, activation='relu',
                                          custom_encoder=None, custom_decoder=None)

    def forward(self):
        pass
