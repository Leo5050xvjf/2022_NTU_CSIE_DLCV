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
import argparse
from datasets.MyDataset import DigitsDataset
from models.MyDANN import MyDANN
import argparse

if __name__ == "__main__":


    #cls = "mnistm"
    # cls = "usps"
    # cls = "svhn"
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_path', default=f"./hw2_data/digits/mnistm/test",
                        help='input directory (target domain image)', type=str)
    parser.add_argument('-c', '--class_name', default="mnistm", help='class name of target domain', type=str)
    parser.add_argument('-o', '--out_path', default=f"./output/hw3/3_3_2/mnistm_pred.csv", help='output directory',type=str)
    args = parser.parse_args()

    input_path = args.in_path
    target_name = args.class_name
    output_path = args.out_path


    '''     Source      Target
    ------------------------------
    R1:     MNISTM     USPS
    R2:     SVHN       MNISTM
    R3:     USPS       SVHN
    '''

    # Setup checkpoint path
    if target_name == "mnistm":
        CHECKPOINT = "./checkpoint/domainMIX_mnistm_25_47.02.pth"
    elif target_name == "svhn":
        CHECKPOINT = "./checkpoint/domainMIX_svhn_10_28.75.pth"
    elif target_name == "usps":
        CHECKPOINT = "./checkpoint/domainMIX_usps_24_78.38.pth"

    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # **********************

    #To_TA = 1
    To_TA = 1



    # **********************

    if To_TA == 1:
    # 給助教時用這個，這個地2個參數是None
        target_test_set = DigitsDataset(input_path, None, t)
    else:
        cls = "mnistm"
        #cls = "usps"
        #cls = "svhn"
     

        target_test_set = DigitsDataset(input_path, f"./hw2_data/digits/{cls}/test.csv", t)

    target_test_loader = DataLoader(target_test_set, batch_size=1, shuffle=False, num_workers=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup Model
    model = MyDANN(domain_open=True).to(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint)

    df = {
        "image_name": [],
        "label": [],
    }

    # Testing
    model.eval()
    correct = 0

    for data, label, file_name in target_test_loader:
        if To_TA == 0:
            label = label.to(device)
            data = data.to(device)
            result, _ = model(data)
            df["image_name"].append(file_name[0])
            df["label"].append(torch.argmax(result).item())
            _,result = torch.max(result,dim = 1)
            correct += 1 if (result == label) else 0

        else:
            data = data.to(device)
            result, _ = model(data)
            df["image_name"].append(file_name[0])
            df["label"].append(torch.argmax(result).item())




    if To_TA == 1:
        df = pd.DataFrame(df)
        df.to_csv(output_path, index=0)
    else:
        df = pd.DataFrame(df)
        df.to_csv(output_path, index=0)
        print(f"Target domain:{cls} , acc: {100. * correct / len(target_test_loader)} %")












