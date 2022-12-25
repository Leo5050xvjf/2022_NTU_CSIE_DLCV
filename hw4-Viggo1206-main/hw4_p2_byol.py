import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision
from yuchin.utils.dataset import MiniDataset,OfficeHomeDataset
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
from torchvision.models import resnet50
from byol_pytorch import BYOL



device = "cuda" if torch.cuda.is_available() else "cpu"

net = resnet50(pretrained=False).to(device)
traDataset = MiniDataset("./hw4_data/mini/train","./hw4_data/mini/train.csv")
trainDataLoader = DataLoader(dataset=traDataset,batch_size=15,shuffle=True)
learner =BYOL(
    net,
    image_size=128,
    hidden_layer = 'avgpool'
).to(device)
opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
epochs = 1000

for epoch in tqdm(range(epochs)):
    total_loss = 0.0
    for i ,(imgs,label) in enumerate(trainDataLoader):
        imgs = imgs.to(device)
        loss = learner(imgs)
        total_loss+=loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average()
    print(f"total_loss is {total_loss} ")
    if (epoch == 1) or (epoch%10 == 0):
        torch.save(net.state_dict(),f"./checkpoints/B/{total_loss}.pth")




