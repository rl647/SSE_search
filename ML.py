#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:07:12 2025

@author: runfeng
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 2), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 2), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Manually adjust the last dimension of identity to match out, if necessary
        if identity.shape != out.shape:
            # Assuming the mismatch is in the last dimension and needs duplication
            identity = identity.repeat(1, 1, 1, 2)[:, :, :, :out.shape[-1]]

        # print("Adjusted identity shape:", identity.shape)
        out += identity
        out = F.relu(out)
        return out

# classification
class HybridResNetBiLSTM(nn.Module):
    def __init__(self):
        super(HybridResNetBiLSTM, self).__init__()
        self.resblock1 = ResNetBlock(2, 16)
        self.resblock2 = ResNetBlock(16, 32)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((None, 1))  # Adaptive pooling
        self.bilstm = nn.LSTM(32, 64, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 1)  # 64 * 2 because of BiLSTM

    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.adaptive_pool(x)
        x = x.squeeze(-1).permute(0, 2, 1)
        x, _ = self.bilstm(x)
        # Using the last hidden state from forward and backward sequences
        x = torch.cat((x[:, -1, :64], x[:, 0, 64:]), dim=1)
        # x = torch.sigmoid(self.fc(x))
        x = self.fc(x)

        return x

# model.to(device)  #
#%%
