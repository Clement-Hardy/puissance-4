# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 21:21:43 2020

@author: Hard Med lenovo
"""
import torch.nn as nn
import torch.nn.functional as F


def model_basic(input_shape, output_shape):
    model = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.Sigmoid(),
            nn.Linear(64, 64),
            nn.Sigmoid(),
            nn.Linear(64, 64),
            nn.Sigmoid(),
            nn.Linear(64, output_shape)
        )
    return model