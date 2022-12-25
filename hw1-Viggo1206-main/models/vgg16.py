
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import datetime
class MyVGG16(nn.Module):
    def __init__(self):
        super(MyVGG16, self).__init__()
        net = torchvision.models.vgg16(pretrained=True)
        self.features = net.features
        # self.avgpool = net.avgpool
        self.classifier =  nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(8192, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 50),
        )

    def forward(self, data):
        out = self.features(data)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out