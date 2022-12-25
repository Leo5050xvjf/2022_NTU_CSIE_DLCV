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

import argparse

import glob
import time

from models.FCN8 import MyFCN8
from models.FCN32 import MyFCN32


cls_color = {
    0:  [0, 255, 255],
    1:  [255, 255, 0],
    2:  [255, 0, 255],
    3:  [0, 255, 0],
    4:  [0, 0, 255],
    5:  [255, 255, 255],
    6: [0, 0, 0],
}

class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.file_names = os.listdir(self.root)
        self.file_pathes = [(os.path.join(self.root, f)) for f in self.file_names]
        self.length = len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        file_path = self.file_pathes[index]

        img = Image.open(file_path)

        if self.transform != None:
            img = self.transform(img)

        return img, file_name

    def __len__(self):
        return self.length


def make_mask(mask):
    h, w = mask.shape[:2]
    result = np.zeros((h, w, 3), dtype='uint8')
    for i in range(7):
        result[mask == i] = np.array(cls_color[i], dtype='uint8')

    return result


def test(model, test_loader, output_path, device="cuda"):
    model.eval()  # Important: set evaluation mode
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for num, (data, file_name) in enumerate(test_loader):
            data = data.to(device)
            output = model(data)

            mask = output[0].cpu().detach().numpy()
            mask = np.argmax(mask, axis=0)
            mask = make_mask(mask)
            Image.fromarray(mask).save(os.path.join(output_path, file_name[0]))

            # if num % 200 == 0:
            #     print(f"testing: {num} / {len(test_loader)}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', default="./validation", help='test data directory', type=str)
    parser.add_argument('-o', '--out', default="./test_file", help='output directory', type=str)
    parser.add_argument('-m', '--model', default="fcn8", help='model name', type=str)
    args = parser.parse_args()
    #  testing_path 是輸入的影像
    # output_path 是輸出的影像->七彩圖，也是助教要測試的地方
    testing_path = args.test
    output_path = args.out


    # 手動強制設定model
    # args.model = "fcn8"
    # model_path = "checkpoint/20211031_193345-fcn8-32.pth"
    # output_path = "./fcn8_32_result"
    # testing_path = "./p2_data/validation"

    # args.model = "fcn32"
    # model_path = "checkpoint/20211031_200147-fcn32-27.pth"
    # output_path = "./fcn32_result"
    # testing_path = "./p2_data/validation"






    model_path = "checkpoint_p2/fcn8_32.pth"
    model = MyFCN8().to("cuda")
    checkpoint = torch.load(model_path, map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'])

    test_set = MyDataset(root=testing_path, transform=transforms.ToTensor())
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

    test(model, test_loader, output_path)




