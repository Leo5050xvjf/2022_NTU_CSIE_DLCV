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
import numpy as np


class FaceDataset(Dataset):
    def __init__(self, root,transform=None):
        self.transform = transform

        # about path
        self.root = root

        # about image info.
        self.filenames = os.listdir(self.root)
        self.filepathes = [os.path.join(self.root, f) for f in self.filenames]
        self.length = len(self.filenames)

        # # about csv info.
        # self.labels = pd.read_csv(self.csv_path).set_index('image_name').T.to_dict('list')

    def __getitem__(self, index):
        img = Image.open(self.filepathes[index])
        file_name = self.filenames[index]
        # label = self.labels[file_name]


        if self.transform != None:
            img = self.transform(img)
            # label = torch.tensor(label)

        return img, file_name

    def __len__(self):
        return self.length




class DigitsDataset(Dataset):
    def __init__(self, root, csv_path=None, transform=None):
        self.transform = transform

        # about path
        self.root = root
        self.csv_path = csv_path

        # about image info.
        self.filenames = sorted(os.listdir(self.root))
        self.filepathes = [os.path.join(self.root, f) for f in self.filenames]
        self.length = len(self.filenames)

        # about csv info.
        # self.labels = pd.read_csv(self.csv_path).set_index('image_name').T.to_dict('index')['label']
        self.labels = pd.read_csv(self.csv_path).set_index('image_name').to_dict()['label'] if self.csv_path != None else "None"

    def __getitem__(self, index):
        img = Image.open(self.filepathes[index]).convert('RGB')
        file_name = self.filenames[index]
        label = self.labels[file_name] if self.labels != "None" else "None"

        if self.transform != None:
            img = self.transform(img)

        return img, label, file_name

    def __len__(self):
        return self.length


if __name__ == "__main__":
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    Data = DigitsDataset("../HW2/yuchin/hw2_data/digits/mnistm/train/","../HW2/yuchin/hw2_data/digits/mnistm/train.csv",t)
    train_loader = DataLoader(Data, batch_size=20, shuffle=True, )
    for img,labels,_ in train_loader:
        print(labels.size())
        input()





















