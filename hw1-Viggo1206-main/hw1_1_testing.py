
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
import argparse
import pandas


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
        return img, self.fileName[index]
    def __len__(self):return self.length
def validate(model,trainData,valData,batchSize = 1):
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
def testing(checkpoint_path  = "checkpoint/100.pth",testDataPath = "p1_data/val_50"):
    # checkpoint "./vgg16_res8/100.pth"
    # testDtatPath './p1_data/val_50'

    d = {
        "image_id": [],
        "label": [],
    }
    batchSize = 1
    model = MyVGG16().to('cuda')
    checkpointFile = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpointFile['state_dict'])
    t = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5077, 0.4813, 0.4312), (0.2627, 0.2547, 0.2736)),
    ])
    valData = MyDataSet(testDataPath, transform=t)
    val_loader = DataLoader(valData,batch_size=batchSize,shuffle=False)
    model.eval()

    with torch.no_grad():
        for name,loader in [("validate",val_loader)]:
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs,img_name in loader:
                    imgs = imgs.to('cuda')
                    # labels = labels.to('cuda')
                    outputs = model(imgs)
                    # predicted 是model 認為的答案
                    _,predicted = torch.max(outputs,dim = 1)
                    predicted= predicted.cpu()
                    predicted = predicted.numpy()[0]


                    d["image_id"].append(img_name[0])
                    d["label"].append(predicted)

                    # total += labels.shape[0]
                    # correct += int((predicted == labels).sum())

    return d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', default="./output_test/hw2_1/testing_50", help='test data directory', type=str)
    parser.add_argument('-o', '--out', default="csv_ans/", help='output directory', type=str)
    args = parser.parse_args()

    testing_path = args.test
    output_path = args.out
    # print(testing_path)
    # print(output_path)

    # model_path = "CheckPoints/vgg16_22.pth"

    checkpoint_path = "./checkpoint_p1/100.pth"
    testData_path = "./p1_data/val_50"
    d=testing(checkpoint_path,testing_path)

    d = pandas.DataFrame(d)
	#, 'test_pred.csv'
    d.to_csv(os.path.join(output_path),index=0)
