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
import math

from datasets.MyDataset import DigitsDataset
from models.MyDANN import MyDANN
from sklearn import manifold

class_color = {
    0: "gray",
    1: "lightcoral",
    2: "coral",
    3: "peachpuff",
    4: "gold",
    5: "yellowgreen",
    6: "forestgreen",
    7: "turquoise",
    8: "cornflowerblue",
    9: "violet",
}

# model = MyDANN(True)
# torch.save(model.state_dict(),"./DANN_checkpoint/test.pth")


# 此model執行2種domain 的風格遷移，因此domain_open設為True
if __name__ == "__main__":

    import datetime
    import time

    task_list = {
        1: ("svhn", "mnistm"),
        2: ("mnistm", "usps"),
        # 3: ("usps", "svhn"),
    }

    for task_num in [1,2]:

        # Hyper Parameters
        SOURCE, TARGET = task_list[task_num]


        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # batch_size = 1024
        batch_size = 200
        lr = 1e-3
        betas = (0.8, 0.999)
        epoch = 30

        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        source_train_set = DigitsDataset(f'./hw2_data/digits/{SOURCE}/train', f'./hw2_data/digits/{SOURCE}/train.csv',t)
        target_train_set = DigitsDataset(f'./hw2_data/digits/{TARGET}/train', f'./hw2_data/digits/{TARGET}/train.csv',t)

        source_test_set  = DigitsDataset(f'./hw2_data/digits/{SOURCE}/test' , f'./hw2_data/digits/{SOURCE}/test.csv', t)
        target_test_set  = DigitsDataset(f'./hw2_data/digits/{TARGET}/test' , f'./hw2_data/digits/{TARGET}/test.csv', t)

        source_train_size = len(source_train_set)
        target_train_size = len(target_train_set)

        # data loader
        # train_size = int(0.8 * source_train_size)
        # validate_size = source_train_size - train_size
        # source_train_set, source_valid_set = torch.utils.data.random_split(source_train_set, [train_size, validate_size])
        source_train_loader = DataLoader(source_train_set, batch_size=batch_size, shuffle=True, num_workers=1)
        # source_valid_loader = DataLoader(source_valid_set, batch_size=batch_size, shuffle=False, num_workers=1)
        source_test_loader = DataLoader(source_test_set, batch_size=batch_size, shuffle=False, num_workers=1)
        source_fix_test = iter(source_test_loader).next()

        # train_size = int(0.8 * target_train_size)
        # validate_size = target_train_size - train_size
        # target_train_set, target_valid_set = torch.utils.data.random_split(target_train_set, [train_size, validate_size])
        target_train_loader = DataLoader(target_train_set, batch_size=batch_size, shuffle=True, num_workers=1)
        # target_valid_loader = DataLoader(target_valid_set, batch_size=batch_size, shuffle=False, num_workers=1)

        target_test_loader = DataLoader(target_test_set, batch_size=batch_size, shuffle=False, num_workers=1)
        target_fix_test = iter(target_test_loader).next()

        # HW 3.3.1~3 -> Train: source, test: target
        model = MyDANN(domain_open=True).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
        # optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
        #
        # criterion_c = nn.CrossEntropyLoss()
        # criterion_s = nn.CrossEntropyLoss()

        criterion_c = torch.nn.NLLLoss()
        criterion_s = torch.nn.NLLLoss()

        for ep in range(1, epoch + 1):
            print(f"This is {ep} epochs!\n")
            t1 = time.time()
            total_train_loss = 0
            total_train_s_class_loss = 0
            total_train_s_domain_loss = 0
            total_train_t_domain_loss = 0
            total_train_correct = 0

            min_len = min(len(source_train_loader), len(target_train_loader))

            # TODO: Training
            model.train()
            for batch_idx, ((s_data, s_label, _), (t_data, t_label, _)) in enumerate(zip(source_train_loader, target_train_loader)):
                p = float(batch_idx + ep * min_len) / epoch / min_len
                Lambda = 2. / (1. + np.exp(-10 * p)) - 1

                s_data, s_label = s_data.to(device), s_label.to(device)
                t_data, t_label = t_data.to(device), t_label.to(device)

                # Train on source training data
                optimizer.zero_grad()
                class_output, domain_output = model(s_data, Lambda=Lambda)

                s_class_loss = criterion_c(class_output, s_label)
                s_domain_loss = criterion_s(
                    domain_output,
                    torch.zeros(len(s_data)).long().to(device)
                )

                # Train on target training data
                _, domain_output = model(t_data, Lambda=Lambda)
                t_domain_loss = criterion_s(
                    domain_output,
                    torch.ones(len(t_data)).long().to(device)
                )

                loss = t_domain_loss + s_domain_loss + s_class_loss
                loss.backward()
                optimizer.step()

                # Compute ACC
                pred = class_output.max(dim=1)[1]
                correct = torch.sum(pred == s_label.view_as(pred)).item()

                total_train_loss += loss.item()
                total_train_s_class_loss += s_class_loss.item()
                total_train_s_domain_loss += s_domain_loss.item()
                total_train_t_domain_loss += t_domain_loss.item()
                total_train_correct += correct
            #     source ({SOURCE}) train - acc:
            print(f"source ({SOURCE}) train - acc:  : {100*total_train_correct / min(len(source_train_loader.dataset), len(target_train_loader.dataset)):.4f}%")


            # TODO: Testing on target_test data
            total_test_loss = 0
            total_test_class_loss = 0
            total_test_domain_loss = 0
            total_test_correct = 0
            model.eval()
            with torch.no_grad():
                for batch_idx, (data, label, _) in enumerate(target_test_loader):
                    data, label = data.to(device), label.to(device)

                    class_output, domain_output = model(data, Lambda=0)

                    pred = class_output.max(dim=1)[1]
                    correct = torch.sum(pred == label.view_as(pred)).item()

                    class_loss = criterion_c(class_output, label)
                    domain_loss = criterion_s(
                        domain_output,
                        torch.ones(len(data)).long().to(device)
                    )

                    total_test_loss += loss.item()
                    total_test_class_loss += class_loss.item()
                    total_test_domain_loss += domain_loss.item()
                    total_test_correct += correct
                acc_is = f"{100. * total_test_correct / len(target_test_loader.dataset): .2f}"
                print(f"##important## target test {TARGET}，testing_traget acc is : {100. * total_test_correct / len(target_test_loader.dataset):.2f}%")
                print("==================================================================")



            # TODO: TSNE Result
            if ep % 5 == 0:

                model.eval()
                with torch.no_grad():
                    s_data, s_label, _ = source_fix_test
                    t_data, t_label, _ = target_fix_test

                    cat_data = torch.cat([s_data, t_data], dim=0).to(device)
                    cat_label = torch.cat([s_label, t_label], dim=0).to(device)

                    latent = model.feature_extractor(cat_data).view(cat_data.size(0), -1).cpu()

                    # t-SNE
                    X_tsne = manifold.TSNE(n_components=2, init='pca', verbose=1, n_iter=1000).fit_transform(latent)

                    # plot scatter figure
                    plt.figure()
                    plt.title(f"class_ep_{ep}")
                    # 總共plot 1028 * 2 個點，每個點用label著色
                    for (x, y), l in zip(X_tsne, cat_label):
                        plt.scatter(x, y, s=10, c=class_color[l.item()], alpha=0.8)
                    #     存結果
                    # 1: ("svhn", "mnistm"),
                    # 2: ("mnistm", "usps"),
                    # 3: ("usps", "svhn"),
                    saveImgFileName = ["SM","MU","US"]
                    # 以source domain分別存檔案
                    plt.savefig(f"./tSNE_img/{saveImgFileName[task_num-1]}/{SOURCE}_source_digit_{ep}.png", dpi=300)
                    plt.close()

                    plt.figure(),
                    plt.title(f"domain_ep_{ep}")

                    # 這邊的意思是，在上一個for迴圈中(x,y)的數量為 min(X_tSNE,cat_label)，而X_tSNE的大小和裡面的data順序是固定的，
                    # 因此在上一層for迴圈中，
                    # 總共plot 1280 * 2 個點，每個點用domain著色
                    for num, (x, y) in enumerate(X_tsne):
                        plt.scatter(x, y, s=10, c="b" if num < batch_size else "r", alpha=0.8)
                    #     存結果
                    plt.savefig(f"./tSNE_img/{saveImgFileName[task_num-1]}/{SOURCE}_source_domainMIX_no_k_fold_{ep}.png" ,dpi=300)
                    plt.close()


            # TODO: Summary for each epoch
            t2 = time.time()

            if ep % 1 == 0:
                torch.save(model.state_dict(),f"./DANN_checkpoint/domainMIX_no_k_fold/domainMIX_{TARGET}_{ep}_{acc_is}.pth")
#                 最後應該產生3種 負責分析各自target的pth，每種pth 各30個


