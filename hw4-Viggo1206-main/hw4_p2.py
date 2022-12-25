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


import csv
import random
import numpy as np
import pandas as pd

'''
         | Pretraind |                  Fine-Tune   |               BYOL
------------------------------------------------------------------------------------
     A   |     X     |                  Train Resnet|            X
------------------------------------------------------------------------------------
     B   |     O     |                  Train Resnet|            X
------------------------------------------------------------------------------------
     C   |     X     |                  Train Resnet|            O
------------------------------------------------------------------------------------
     D   |     O     |Train Resnet (only classifier)|            X
------------------------------------------------------------------------------------
     E   |     X     |Train Resnet (only classifier)|            O
------------------------------------------------------------------------------------

'''
def getDict():
    strToNum = {'Couch': 0, 'Helmet': 1, 'Refrigerator': 2, 'Alarm_Clock': 3, 'Bike': 4, 'Bottle': 5,
                     'Calculator': 6, 'Chair': 7, 'Mouse': 8, 'Monitor': 9, 'Table': 10, 'Pen': 11, 'Pencil': 12,
                     'Flowers': 13, 'Shelf': 14, 'Laptop': 15, 'Speaker': 16, 'Sneakers': 17, 'Printer': 18,
                     'Calendar': 19, 'Bed': 20, 'Knives': 21, 'Backpack': 22, 'Paper_Clip': 23, 'Candles': 24,
                     'Soda': 25, 'Clipboards': 26, 'Fork': 27, 'Exit_Sign': 28, 'Lamp_Shade': 29, 'Trash_Can': 30,
                     'Computer': 31, 'Scissors': 32, 'Webcam': 33, 'Sink': 34, 'Postit_Notes': 35, 'Glasses': 36,
                     'File_Cabinet': 37, 'Radio': 38, 'Bucket': 39, 'Drill': 40, 'Desk_Lamp': 41, 'Toys': 42,
                     'Keyboard': 43, 'Notebook': 44, 'Ruler': 45, 'ToothBrush': 46, 'Mop': 47, 'Flipflops': 48,
                     'Oven': 49, 'TV': 50, 'Eraser': 51, 'Telephone': 52, 'Kettle': 53, 'Curtains': 54, 'Mug': 55,
                     'Fan': 56, 'Push_Pin': 57, 'Batteries': 58, 'Pan': 59, 'Marker': 60, 'Spoon': 61,
                     'Screwdriver': 62, 'Hammer': 63, 'Folder': 64}
    numToStr = {0: 'Couch', 1: 'Helmet', 2: 'Refrigerator', 3: 'Alarm_Clock', 4: 'Bike', 5: 'Bottle',
                     6: 'Calculator', 7: 'Chair', 8: 'Mouse', 9: 'Monitor', 10: 'Table', 11: 'Pen', 12: 'Pencil',
                     13: 'Flowers', 14: 'Shelf', 15: 'Laptop', 16: 'Speaker', 17: 'Sneakers', 18: 'Printer',
                     19: 'Calendar', 20: 'Bed', 21: 'Knives', 22: 'Backpack', 23: 'Paper_Clip', 24: 'Candles',
                     25: 'Soda', 26: 'Clipboards', 27: 'Fork', 28: 'Exit_Sign', 29: 'Lamp_Shade', 30: 'Trash_Can',
                     31: 'Computer', 32: 'Scissors', 33: 'Webcam', 34: 'Sink', 35: 'Postit_Notes', 36: 'Glasses',
                     37: 'File_Cabinet', 38: 'Radio', 39: 'Bucket', 40: 'Drill', 41: 'Desk_Lamp', 42: 'Toys',
                     43: 'Keyboard', 44: 'Notebook', 45: 'Ruler', 46: 'ToothBrush', 47: 'Mop', 48: 'Flipflops',
                     49: 'Oven', 50: 'TV', 51: 'Eraser', 52: 'Telephone', 53: 'Kettle', 54: 'Curtains', 55: 'Mug',
                     56: 'Fan', 57: 'Push_Pin', 58: 'Batteries', 59: 'Pan', 60: 'Marker', 61: 'Spoon',
                     62: 'Screwdriver', 63: 'Hammer', 64: 'Folder'}
    return numToStr,strToNum
def only_train_classifier(m):
    if isinstance(m,nn.Conv2d):
        m.requires_grad_(False)


# model setting

device = "cuda" if torch.cuda.is_available() else "cpu"


net = resnet50(pretrained=False)
# train backbone or not
net.apply(only_train_classifier)

# load TA pth or BYOL pth or not
net.load_state_dict(torch.load("./hw4_data/pretrain_model_SL.pt"))
checkpoint = torch.load("./checkpoints/C_pretrain_ep400.pth", map_location=device)
net.load_state_dict(checkpoint['state_dict'])


# checkpoint = torch.load("./checkpoints/C_pretrain_ep400.pth", map_location=device)
# net.load_state_dict(checkpoint)
net.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 65),
        )


traDataset = OfficeHomeDataset("./hw4_data/office/train","./hw4_data/office/train.csv")
trainDataLoader = DataLoader(dataset=traDataset,batch_size=64,shuffle=True)

valDataset = OfficeHomeDataset("./hw4_data/office/val","./hw4_data/office/val.csv")
valDataLoader = DataLoader(dataset=valDataset,batch_size=64,shuffle=False)

loss_fn = nn.CrossEntropyLoss().to(device)
epochs = 100
lr = 0.001
params = [
    {'params':net.conv1.parameters(),'lr':lr/10},
    {'params':net.bn1.parameters(),'lr':lr/10},
    {'params':net.layer1.parameters(),'lr':lr/8},
    {'params':net.layer2.parameters(),'lr':lr/6},
    {'params':net.layer3.parameters(),'lr':lr/4},
    {'params':net.layer4.parameters(),'lr':lr/2},
    {'params':net.fc.parameters(),'lr':lr/10},
]
optimizer = optim.Adam(params,lr = lr)


net.to(device)
max_acc = 0
for epoch in tqdm(range(1,epochs+1)):
    total_loss = 0.0
    net.train()
    for i,(img,_,labelNum) in enumerate(trainDataLoader):
        optimizer.zero_grad()
        img = img.to(device = device)
        labelNum = labelNum.to(device = device)
        pred = net(img)

        loss = loss_fn(pred, labelNum)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()



#     valdate
    net.eval()
    with torch.no_grad():
        for name,loader in [("train",trainDataLoader),("validate",valDataLoader)]:
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs,_,labelNum in loader:
                    imgs = imgs.to(device)
                    labelNum = labelNum.to(device)
                    pred = net(imgs)
                    _,predicted = torch.max(pred,dim = 1)
                    total += labelNum.shape[0]
                    correct += int((predicted == labelNum).sum())
            if (correct/total) > max_acc:
                max_acc = correct/total
                PATH = f"./checkpoints/E/{epoch}.pth"
                torch.save(net.state_dict(), PATH)
            print(f"total_loss is {total_loss}")
            print("{}:Accuracy : {:.2f}".format(name,correct/total))















