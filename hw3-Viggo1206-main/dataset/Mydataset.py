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



class MyDataSet():
    def __init__(self,root,transform = None):
        self.transform = transform
        self.FileNames = os.listdir(root)
        self.Labels = [int(label.split("_")[0]) for label in self.FileNames]
        self.ALLFilepath = [os.path.join(root,filename) for filename in self.FileNames]
        self.length = len(self.FileNames)

    def __getitem__(self, index):
        img = Image.open(self.ALLFilepath[index]).convert('RGB')
        img_Label = self.Labels[index]
        image_name = self.FileNames[index]
        if self.transform  != None:
            img =self.transform (img)
        return img,img_Label,image_name

    def __len__(self): return self.length
def My_collect(batch):
    imgs = [item[0] for item in batch]
    img_labels = [item[1] for item in batch]
    img_name = [item[2] for item in batch]
    return imgs,img_labels,img_name

if __name__ == '__main__':

    t = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])



    T = MyDataSet("../hw3_data/p1_data/train",t)
    # print(T.length)
    tra_dataloader = DataLoader(T,batch_size=5,shuffle=False)
    for img,img_labels,img_name in tra_dataloader:
        for _ in img:

            _ = _.permute(1, 2, 0)
            # _ = _.numpy().astype("uint8")
            print(_.shape)
            print(type(_))
            # print(_)
            # input()
            plt.imshow(_)
            plt.show()

    # trainiter = iter(tra_dataloader)
    # imgs, labels,img_names = trainiter.next()
    #
    # print(f"labels is {labels}")
    # print(f"img_name is {img_names}")
    # for _ in imgs:
    #     print(_.shape)
    # for img,img_labels,img_name in tra_dataloader:
    #     print(img_labels)
    #     print(type(img_labels))
    #     print(img_labels.shape)
    #     input()

