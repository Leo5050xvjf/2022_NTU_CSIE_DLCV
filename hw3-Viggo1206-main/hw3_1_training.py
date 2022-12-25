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
from pytorch_pretrained_vit import ViT
from HW3.dataset.Mydataset import MyDataSet,My_collect

from timm import create_model
from timm.optim import create_optimizer_v2, optimizer_kwargs
from HW3.P1.sam import SAM

# model_name = 'B_16_imagenet1k'
# model = ViT(model_name, pretrained=True)

model_name = "vit_base_patch16_224"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device = ", device)
# create a ViT model : https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py


def show_similarity(model):
    pos_embed = model.pos_embed
    print(pos_embed.shape)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle("Visualization of position embedding similarities", fontsize=24)
    for i in range(1, pos_embed.shape[1]):
        sim = F.cosine_similarity(pos_embed[0, i:i+1], pos_embed[0, 1:], dim=1)
        sim = sim.reshape((14, 14)).detach().cpu().numpy()
        ax = fig.add_subplot(14, 14, i)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.imshow(sim)
    plt.show()




t = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

tra_root = "../hw3_data/p1_data/train"
tra_dataset = MyDataSet(tra_root,t)
# tra_dataset,a = torch.utils.data.random_split(tra_dataset,[100,len(tra_dataset)-100])
tra_dataloader = DataLoader(tra_dataset,batch_size=10,shuffle=True)

val_root = "../hw3_data/p1_data/val"
val_dataset = MyDataSet(val_root,t)
val_dataloader = DataLoader(val_dataset,batch_size=1,shuffle=False)



model = create_model(model_name, pretrained=True, num_classes=37).to(device)
device = "cuda" if torch.cuda.is_available() else "cpu"
loss_func = nn.CrossEntropyLoss()
base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
optimizer = SAM(model.parameters(), base_optimizer, lr=0.001, momentum=0.9)

epochs = 500

for ep in range(1,epochs+1):
    print(f"ep = {ep}")
    model.train()
    for img, label,img_name in tra_dataloader:

        # label = torch.tensor(label,dtype=torch.int64)
        img = img.to(device=device)
        label = label.to(device = device).view(-1)
        # first forward-backward pass
        optimizer.zero_grad()
        pred = model(img)


        loss = loss_func(pred,label)  # use this loss for any training statistics
        # print(loss.item())

        loss.backward()
        optimizer.first_step(zero_grad=True)
        # second forward-backward pass
        pred2 = model(img)
        loss = loss_func(pred2 ,label)
        loss.backward()  # make sure to do a full forward pass
        optimizer.second_step(zero_grad=True)


    print("=========================validating!====================================")

    with torch.no_grad():
        model.eval()
        correct = 0
        for val_img, val_label,val_img_name in val_dataloader:
            val_img = val_img.to(device=device)
            val_label = val_label.to(device=device).view(-1)
            pred = model(val_img)
            _,pred = torch.max(pred,dim=1)
            correct+=sum(pred == val_label)
        print(f"epoch is {ep} , acc is {correct/len(val_dataloader)}")
    if (ep % 5) == 0:
        print("saving!")
        torch.save(model.state_dict(),f"./P1_checkpoint/{ep}.pth")







