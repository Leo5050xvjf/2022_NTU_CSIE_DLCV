import os
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
import pandas as pd


def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class Convnet(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
        super(Convnet, self).__init__()
        self.encoder = nn.Sequential(
            # in: 3 x 84 x 84
            conv_block(in_channels, hid_channels),
            # out: 64 x 42 x 42
            conv_block(hid_channels, hid_channels),
            # out: 64 x 21 x 21
            conv_block(hid_channels, hid_channels),
            # out: 64 x 10 x 10
            conv_block(hid_channels, out_channels),
            # out: 64 x 5 x 5 -> 1600
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


if __name__ == "__main__":
    x = torch.Tensor(100, 3, 84, 84)
    model = Convnet()
    y = model(x)
    print(model)
    # print(f"x.size = {x.size()}")
    # print(f"y.size = {y.size()}")
