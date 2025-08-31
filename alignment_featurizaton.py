#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:10:38 2025

@author: runfeng
"""

import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import string
import random
#%%
element = {}
osc = {}

osc_helix = {}
osc_sheet = {}
H = list(string.ascii_lowercase)
E = list(string.ascii_uppercase)

# Initialize dictionaries
for i, (h, e) in enumerate(zip(H, E)):
    if i <= 8:
        osc_helix[h] = (i * 1) + 2
        osc_sheet[e] = i + 2
    elif i > 8 and i <= 18:
        osc_helix[h] = (i * 2) - 6
        osc_sheet[e] = i * 2 - 6
    elif i > 18:
        osc_helix[h] = (i * 3) - 24
        osc_sheet[e] = i * 3 - 24

#%%

# featurized_input = {}
def alignment_to_features(seq1, seq2, osc_helix, osc_sheet):
    length = max(len(seq1), len(seq2))
    # Adjusting depth to 2 to represent helix and sheet scores separately
    features = np.zeros((length, 2, 3), dtype=np.int16)  # Height, Width, Depth(2 for helix and sheet)
    seq1 = list(filter(None,(seq1.strip().split('_'))))[:]
    seq2 = list(filter(None,(seq2.strip().split('_'))))[:]
    # print(seq1,seq2)
    for i, (s1, s2) in enumerate(zip(seq1, seq2)):
        # print(i,s1,s2)
        
        # Assign scores from osc_helix or osc_sheet based on character
        if (s1.islower() and s2.islower()) or (s1.islower() and s2=='-') or (s1=='-' and s2.islower()):
            features[i, 0, 0] = osc_helix.get(s1, -1)
            features[i, 1, 0] = osc_helix.get(s2, -1)
        elif s1.islower() and s2.isupper():
            features[i, 0, 0] = osc_helix.get(s1, -1)
            features[i, 1, 1] = osc_sheet.get(s2, -1)
        elif s1.isupper() and s2.islower():
            features[i, 0, 1] = osc_sheet.get(s1, -1)
            features[i, 1, 0] = osc_helix.get(s2, -1)
        elif (s1.isupper() and s2=='-') or (s1=='-' and s2.isupper()) or (s1.isupper() and s2.isupper()):
            features[i, 0, 1] = osc_sheet.get(s1, -1)
            features[i, 1, 1] = osc_sheet.get(s2, -1)
        else:
            features[i, 0, 2] = int(s1) if s1!='-' else -1
            features[i, 1, 2] = int(s2) if s2!='-' else -1
    features = np.transpose(features, (1, 0, 2))
    return features

#%%
import torch
import torch.nn.functional as F
def custom_pad_sequences(batch):
    # print(len(batch[0]))
    max_length = max([s.size(1) for s,_ in batch])

    padded_sequences = []
    ori_labels = []
    for sequence, ori_label in batch:
        # Calculate padding length for the sequence_length dimension
        padding_length = max_length - sequence.size(1)
        
        # Apply padding
        if padding_length > 0:
            # Pad on the sequence_length dimension
            padded_sequence = F.pad(sequence, (0, 0, 0, padding_length), 'constant', 0)
        else:
            padded_sequence = sequence
        
        padded_sequences.append(padded_sequence)
        ori_labels.append(ori_label)
    
    # Stack or concatenate tensors
    padded_sequences = torch.stack(padded_sequences)
    return padded_sequences, ori_labels
from torch.utils.data import Dataset, DataLoader, random_split

class SequenceAlignmentDataset(Dataset):
    def __init__(self, alignments,ori_TM):
        """
        alignments: List of featurized alignments (as tensors)
        labels: List of labels (0 or 1)
        """
        self.alignments = alignments
        self.ori_label = ori_TM

    def __len__(self):
        return len(self.alignments)

    def __getitem__(self, idx):
        return self.alignments[idx],  self.ori_label[idx]


def evaluate(model, test_loader, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    predictions = []
    all_labels = []
    actual = []
    with torch.no_grad():
        for alignments, ori_label in test_loader:
            alignments = alignments.to(device)
    
            outputs = model(alignments)
    
            predictions+=list(outputs.detach().cpu().numpy())
            actual+=list(ori_label)
    
    
       
    return  predictions,actual
