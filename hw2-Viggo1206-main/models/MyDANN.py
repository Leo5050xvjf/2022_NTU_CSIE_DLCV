import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Lambda):
        ctx.Lambda = Lambda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.Lambda
        return output, None


class MyDANN(nn.Module):
    def __init__(self, domain_open: bool):
        '''

        :param domain_open:
            True:   DANN
            False:  Traditional CNN
        '''
        super(MyDANN, self).__init__()

        self.domain_open = domain_open

        self.feature_extractor = nn.Sequential(
            # in: 3 x 28 x 28

            nn.Conv2d(3, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(True),
            # out: 32 x 12 x 12

            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            # out: 64 x 4 x 4
        )

        self.class_predictor = nn.Sequential(
            nn.Linear(64 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout2d(),

            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),

            nn.Linear(100, 10),
            nn.LogSoftmax(dim=1),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),

            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, inputs, Lambda=None):
        features = self.feature_extractor(inputs)
        features = features.view(-1, 64 * 4 * 4)
        class_output = self.class_predictor(features)

        if self.domain_open:
            re_features = GRL.apply(features, Lambda)
            domain_output = self.domain_classifier(re_features)
            return class_output, domain_output
        else:
            return class_output


if __name__ == "__main__":
    model = MyDANN(True)
    print(model)