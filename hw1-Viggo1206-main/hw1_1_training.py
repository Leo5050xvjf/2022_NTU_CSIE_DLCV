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
from models.vgg16 import MyVGG16

class MyDataSet(Dataset):
    def __init__(self,path,transform =None):
        self.transform = transform
        # path = "./p1_data/train_50" or "./p1_data/val_50"
        self.fileName = os.listdir(path)
        self.LabelAndImgPath = [(path+'/'+fileName  ,int(fileName.split('_')[0]) ) for fileName in self.fileName]
        self.length = len(self.LabelAndImgPath)

    def __getitem__(self, index):
        img_name, label = self.LabelAndImgPath[index]
        img = Image.open(img_name)
        if self.transform != None:
            img = self.transform(img)
        # #     輸出的img 已經轉成張量，且label是一個整數
        return img, label
    def __len__(self):return self.length

def train():
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    t = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5077, 0.4813, 0.4312), (0.2627, 0.2547, 0.2736)),
    ])
    dataset = MyDataSet('./p1_data/train_50', transform=t)
    train_loader = DataLoader(dataset = dataset, batch_size=64, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    model = MyVGG16()
    # 把pre-train 鎖死不更新，但結果只有0.72，反而較差
    # for n,p in model.named_parameters():
    #     if n.split(".")[0] == "features":
    #         p.requires_grad_(False)
    model.to(device)

    # optimizer = torch.optim.Adam(params=model.parameters())

    lr = 0.0001
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # 開始訓練
    training_loop_VGG16(100,optimizer,model,loss_fn,train_loader,device)
def training_loop_VGG16(n_epochs,optimizer,model,loss_fn,train_loader,device):
    model.train()
    # print(model)
    for epoch in range(1,n_epochs+1):
        loss_train = 0.0
        print(f"this is the {epoch} epoch!")
        for imgs,labels in train_loader:
            imgs = imgs.to(device = device)
            labels = labels.to(device = device)
            outputs = model(imgs)
            loss = loss_fn(outputs,labels)
            # print(loss)
            # l2_lambda = 0.001
            # le_norm = sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        if epoch == 1 or epoch%10 == 0:
            checkpoint(f"./vgg16_res8/{epoch}.pth", model, optimizer)
        print("Loss : %f"%(float(loss)))
        print(f"{datetime.datetime.now()} Epoch {epoch} , Training loss {loss_train/len(train_loader)}")
    print(f"batch size  is {train_loader.batch_size}")
batchSize = 1
def validate(model,trainData,valData):
    val_loader = DataLoader(valData,batch_size=batchSize,shuffle=False)
    train_loader = DataLoader(trainData,batch_size=batchSize,shuffle=False)
    model.eval()

    with torch.no_grad():
        for name,loader in [("train",train_loader),("validate",val_loader)]:
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs,labels in loader:
                    imgs = imgs.to('cuda')
                    labels = labels.to('cuda')
                    outputs = model(imgs)
                    # print(outputs)
                    # input()
                    _,predicted = torch.max(outputs,dim = 1)
                    total += labels.shape[0]
                    # print(f"predicted is : {predicted}")
                    # print(f"labels is : {labels}")
                    # input()
                    correct += int((predicted == labels).sum())

            print("Accuracy {}: {:.2f}".format(name,correct/total))
def run_validate():
    #開始測試

    num = [1,10,20,30,40,50,60,70,80,90,100]
    num = [100]
    for _ in num:
        model = MyVGG16().to('cuda')
        checkpointFile = torch.load("./vgg16_res8/%d.pth"%(_), map_location='cuda')
        model.load_state_dict(checkpointFile['state_dict'])
        # model.load_state_dict(checkpointFile['optimizer'])

        t = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5077, 0.4813, 0.4312), (0.2627, 0.2547, 0.2736)),
        ])
        trainData = MyDataSet('./p1_data/train_50', transform=t)
        valData = MyDataSet('./p1_data/val_50', transform=t)
        validate(model,trainData,valData)
def checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

if __name__ == "__main__":



    #開始訓練
    # train()
    #多筆pth檔測試
    run_validate()






