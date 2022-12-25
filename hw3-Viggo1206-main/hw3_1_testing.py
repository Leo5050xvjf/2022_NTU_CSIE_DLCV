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
from dataset.Mydataset import MyDataSet,My_collect

from timm import create_model
from timm.optim import create_optimizer_v2, optimizer_kwargs
from sam import SAM
import sys
from PIL import Image
import cv2

import argparse
import csv
import pandas as pd


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model_name = "vit_base_patch16_224"
model = create_model(model_name=model_name,pretrained=False,num_classes=37).to(device)
pth_path = "./25_0.95.pth"
model.load_state_dict(torch.load(pth_path))

t = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# val_root = "../hw3_data/p1_data/val"
# val_dataset = MyDataSet(val_root,t)
# val_dataloader = DataLoader(val_dataset,batch_size=1,shuffle=False)


def pred(val_dataloader,outpath,ToTa = 0):
    if not ToTa:
        with torch.no_grad():
            model.eval()
            correct = 0
            for val_img, val_label, val_img_name in val_dataloader:

                val_img = val_img.to(device=device)
                val_label = val_label.to(device=device).view(-1)
                pred_label = model(val_img)
                _, pred_label = torch.max(pred_label, dim=1)
                print(val_img_name)
                print(pred_label.item())
                input()
                correct += sum(pred_label == val_label)
            print(f" acc is {correct / len(val_dataloader)}")
    else:

        df = {
            "image_id": [],
            "label": [],
        }
        with torch.no_grad():
            model.eval()
            correct = 0
            for val_img, val_label, val_img_name in val_dataloader:
                val_img = val_img.to(device=device)
                val_label = val_label.to(device=device).view(-1)
                pred_label = model(val_img)
                _, pred_label = torch.max(pred_label, dim=1)

                df["image_id"].append(val_img_name[0])
                df["label"].append(pred_label.item())

                correct += sum(pred_label == val_label)
            df = pd.DataFrame(df)
            df.to_csv(outpath, index=0)
            print(f" acc is {correct / len(val_dataloader)}")
def showAttentionMap():

    for _ in range(3):
        image_path = ["../hw3_data/p1_data/val/26_5064.jpg","../hw3_data/p1_data/val/29_4718.jpg","../hw3_data/p1_data/val/31_4838.jpg"]
        img = Image.open(image_path[_])
        img_name = image_path[_].split("/")[-1]
        # print(img_name)
        # input()

        img_t = t(img)
        img_t = torch.unsqueeze(img_t,dim = 0).to(device)
        pred_ = model(img_t)
        patches = model.patch_embed(img_t)
        pos_embed = model.pos_embed
        transformer_input = torch.cat((model.cls_token, patches), dim=1) + pos_embed
        attention = model.blocks[11].attn
        transformer_input_expanded = attention.qkv(transformer_input)[0]
        qkv = transformer_input_expanded.reshape(197, 3, 12, 64)
        q = qkv[:, 0].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
        k = qkv[:, 1].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
        kT = k.permute(0, 2, 1)
        attention_matrix = q @ kT

        def show_mean_and_all_head(img,attention_matrix):
            fig = plt.figure(figsize=(16, 6))
            resize_ = torchvision.transforms.Resize((224,224))
            img = resize_(img)
            ax = fig.add_subplot(2, 7, 1)
            plt.xticks([])
            plt.yticks([])
            img = np.asarray(img)
            plt.imshow(img)
            img_num = 1

            average_att = np.zeros((224,224))
            for _ in range(12):

                if (_ >= 6):
                    img_num = _+3
                else:
                    img_num =_+2

                single_depth_map = attention_matrix[_,:,1:].to("cpu").detach().numpy()
                single_depth_map = np.sum(single_depth_map,axis=0)/197
                single_depth_map = single_depth_map.reshape(14, 14)
                single_depth_map = cv2.resize(single_depth_map,(224,224))
                average_att+=single_depth_map

                ax = fig.add_subplot(2, 7, img_num)
                plt.title(f"attention map {_}")
                plt.xticks([])
                plt.yticks([])
                plt.imshow(single_depth_map)

            average_att = average_att / 12
            ax = fig.add_subplot(2, 7, 8)
            plt.title(f"mean attention map ")
            plt.xticks([])
            plt.yticks([])
            plt.imshow(average_att)
            plt.savefig(f"./{img_name}")
        def show_mean(img,attention_matrix):
            fig = plt.figure(figsize=(16, 6))
            resize_ = torchvision.transforms.Resize((224, 224))
            img = resize_(img)
            ax = fig.add_subplot(1, 2, 1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img)

            average_att = np.zeros((224, 224))
            for _ in range(12):

                single_depth_map = attention_matrix[_,:,1:].to("cpu").detach().numpy()
                single_depth_map = np.sum(single_depth_map,axis=0)/197
                single_depth_map = single_depth_map.reshape(14, 14)
                single_depth_map = cv2.resize(single_depth_map,(224,224))
                average_att+=single_depth_map
            average_att = average_att / 12
            ax = fig.add_subplot(1, 2, 2)
            plt.title(f"mean attention map ")
            plt.xticks([])
            plt.yticks([])
            plt.imshow(average_att)
            plt.savefig(f"./{img_name}")
        show_mean(img,attention_matrix)


    # show depth map
    if 0:
        attn_heatmap_all = np.zeros((196, 224, 224))
        for i in range(196):
            attn_heatmap = attention_matrix[:, i, 1:].reshape((12,14, 14)).detach().cpu().numpy()

            attn_heatmap_resize = np.zeros((12,224,224))
            for _ in range(12):
                attn_heatmap_resize[_] = cv2.resize(attn_heatmap[_],(224,224))

            attn_heatmap_resize = np.sum(attn_heatmap_resize,axis = 0) / 12
            attn_heatmap_all[i] = attn_heatmap_resize
        attn_heatmap_all = np.sum(attn_heatmap_all, axis=0) / 196
        plt.imshow(attn_heatmap_all)
        plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Image Ateention')
    parser.add_argument('-inputpath', type=str, help='input path to image',default= "../hw3_data/p1_data/val")
    parser.add_argument('-output', type=str, help='output path to image', default='./pred_.csv')
    args = parser.parse_args()
    root = args.inputpath
    csv_output = args.output
    img_names = os.listdir(root)
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # val_root = "../hw3_data/p1_data/val"
    val_dataset = MyDataSet(root, t)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    pred(val_dataloader,csv_output,ToTa=1)

    # show and save att map
    # showAttentionMap()

