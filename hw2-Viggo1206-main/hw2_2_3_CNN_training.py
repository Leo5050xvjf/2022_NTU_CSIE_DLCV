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


if __name__ == "__main__":

    # lower bound
    task_list = {
        1: ("svhn", "mnistm"),
        2: ("mnistm", "usps"),
        3: ("usps", "svhn"),
    }

    # upperbound
    # task_list = {
    #     1: ("mnistm", "mnistm"),
    #     2: ("usps", "usps"),
    #     3: ("svhn", "svhn"),
    # }

    for task_num in range(1,4):

        # Hyper Parameters
        SOURCE, TARGET = task_list[task_num]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size = 1024
        lr = 1e-3
        epoch = 10

        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        source_train_set = DigitsDataset(f'./hw2_data/digits/{SOURCE}/train', f'./hw2_data/digits/{SOURCE}/train.csv', t)
        source_test_set = DigitsDataset(f'./hw2_data/digits/{SOURCE}/test', f'./hw2_data/digits/{SOURCE}/test.csv', t)
        target_train_set = DigitsDataset(f'./hw2_data/digits/{TARGET}/train', f'./hw2_data/digits/{TARGET}/train.csv', t)
        target_test_set = DigitsDataset(f'./hw2_data/digits/{TARGET}/test', f'./hw2_data/digits/{TARGET}/test.csv', t)

        source_train_size = len(source_train_set)
        target_train_size = len(target_train_set)

        source_train_loader = DataLoader(source_train_set, batch_size=batch_size, shuffle=True, num_workers=1)
        source_test_loader = DataLoader(source_test_set, batch_size=batch_size, shuffle=False, num_workers=1)

        target_train_loader = DataLoader(target_train_set, batch_size=batch_size, shuffle=True, num_workers=1)
        target_test_loader = DataLoader(target_test_set, batch_size=batch_size, shuffle=False, num_workers=1)

        # HW 3.3.1 -> Train: source, test: target
        model = MyDANN(domain_open=False).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        criterion = nn.NLLLoss()

        for ep in range(1, epoch+1):
            print(f"this is {ep} epoch !")
            total_train_loss = 0
            total_train_correct = 0

            # TODO: Training on source_train data
            model.train()
            for batch_idx, (data, label, _) in enumerate(source_train_loader):
                data, label = data.to(device), label.to(device)
                optimizer.zero_grad()

                output = model(data)

                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                pred = output.max(dim=1)[1]
                correct = torch.sum(pred == label.view_as(pred)).item()

                total_train_loss += loss.item()
                total_train_correct += correct

            # TODO: Testing on target_test data
            total_test_loss = 0
            total_test_correct = 0
            model.eval()
            with torch.no_grad():
                for batch_idx, (data, label, _) in enumerate(target_test_loader):
                    data, label = data.to(device), label.to(device)

                    output = model(data)

                    loss = criterion(output, label)

                    pred = output.max(dim=1)[1]
                    correct = torch.sum(pred == label.view_as(pred)).item()

                    total_test_loss += loss.item()
                    total_test_correct += correct

            acc_is = f"{total_test_correct / len(target_test_loader.dataset):.4f}"
            print(f'## source : {SOURCE}, target ({TARGET}) test - acc: {total_test_correct / len(target_test_loader.dataset):.4f}')

            if ep % 1 == 0:
                if task_num == 1:
                    file_name ="MM"
                elif task_num == 2:
                    file_name = "UU"
                else:file_name ="SS"

                # torch.save(model.state_dict(),f"./DANN_checkpoint/domain_no_MIX_no_k_fold/lowerbound/{file_name}/domain_no_MIX_{TARGET}_{ep}_{acc_is}.pth")





