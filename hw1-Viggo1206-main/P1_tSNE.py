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
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

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
            # nn.Dropout(0.5),
            # nn.Linear(64, 50),
        )

    def forward(self, data):
        out = self.features(data)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


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

net = MyVGG16()
checkpointFile = torch.load("./vgg16_res8/100.pth" , map_location='cuda')
net.load_state_dict(checkpointFile['state_dict'],strict=False)
net.to('cuda')
net.eval()

t = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5077, 0.4813, 0.4312), (0.2627, 0.2547, 0.2736)),
])
dataset = MyDataSet('./p1_data/val_50', transform=t)
output_size = 64
counter = 0
data=  np.zeros((2500,output_size))
data_label = np.zeros((2500,1))
with torch.no_grad():
    val_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    for img,label in val_loader:
        img = img.to(device = 'cuda')
        label = label.numpy().reshape(1,1)
        left = counter*1
        data_label[left:left+1] = label
        out = net(img)
        out = out.cpu()
        out=  out.numpy()

        data[left:left+1] = out
        counter+=1
data_label.astype(int)
data_label = data_label.reshape(-1)
cmap = plt.cm.get_cmap('hsv', output_size)
X_tsne = manifold.TSNE(n_components=2, init="random", random_state=5, verbose=1).fit_transform(data)
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne -x_min) / (x_max -x_min)  #Normalize

plt.figure(figsize=(16,16))
counter = 0
for i in range(2500):
    counter+=1
    c=  cmap(int(data_label[i]))[:3]
    plt.text(X_norm[i, 0], X_norm[i, 1], str(int(data_label[i])), color=c,
             fontdict={"weight": "bold", "size": 9})
plt.xticks([])
plt.yticks([])
plt.show()