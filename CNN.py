import os
import librosa
import torchaudio
import torch.nn as nn
import torch
import numpy as np
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import glob
import random

class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)         
        self.pool = nn.MaxPool2d(2, 2)                      
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)        
        
        self.fc1 = nn.Linear(20176, 120)
        self.fc2 = nn.Linear(120, num_classes)
        
        #Adicion Dropout
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = torch.flatten(x, 1)               
        x = F.relu(self.fc1(x))
        
        #Adicion Dropout
        x = self.dropout(x)   
                    
        x = self.fc2(x)                       
        return x