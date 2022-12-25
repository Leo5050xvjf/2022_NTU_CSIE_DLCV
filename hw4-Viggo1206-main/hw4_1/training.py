import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from yuchin.utils.dataset import MiniDataset
from yuchin.utils.samplers import GeneratorSampler
from yuchin.utils.util import Similarity, count_acc
from yuchin.utils.vis import Logger, Recoder
from yuchin.models.convnet import Convnet
# from torch.utils.tensorboard import SummaryWriter

note = '''
V1.

SGD with mo
=====================================================
'''



def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N-way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N-shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N-query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--test_csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str, help="Testing images directory")
    parser.add_argument('--testcase_csv', type=str, help="Test case csv")
    parser.add_argument('--output_csv', type=str, help="Output filename")

    return parser.parse_args()

if __name__ == '__main__':
    # Hyper parameters
    n_batch = 600
    n_batch_val = 50
    lr = 0.001
    epoch = 300

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data loader
    train_set = MiniDataset('../hw4_data/mini/train', '../hw4_data/mini/train.csv')
    N_way = 5
    N_shot = 1
    N_query = 15
    sampler = GeneratorSampler(n_batch, N_way, N_shot + N_query, cls_num=64)
    train_loader = DataLoader(dataset=train_set, batch_sampler=sampler)

    valid_set = MiniDataset('../hw4_data/mini/val', '../hw4_data/mini/val.csv')
    sampler = GeneratorSampler(n_batch_val, N_way, N_shot + N_query, cls_num=16)
    valid_loader = DataLoader(dataset=valid_set, batch_sampler=sampler)

    # Model & Optimizer
    model = Convnet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Training
    for ep in range(1, epoch+1):
        print(ep)
        model.train()
        episodic_acc = []
        total_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()

            data = data.to(device)
            p = N_way * N_shot
            data_support, data_query = data[:p], data[p:]

            proto = model(data_support) # -> size: (way * shot, 1600)
            proto = proto.reshape(N_shot, N_way, -1).mean(dim=0) # -> size: (way, 1600)
            label = torch.arange(N_way).repeat(N_query).long().to(device)
            logits = Similarity.euclidean_distance(model(data_query), proto)
            loss = nn.CrossEntropyLoss()(logits, label)
            train_acc = count_acc(logits, label)

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            episodic_acc.append(train_acc)

        average_loss = total_loss/n_batch
        episodic_acc = np.array(episodic_acc)
        acc_mean = episodic_acc.mean()
        acc_std = episodic_acc.std()

        # Validation
        model.eval()
        episodic_acc = []
        total_loss = 0

        # t1 = time.time()
        for batch_idx, (data, _) in enumerate(valid_loader):
            data = data.to(device)
            p = N_way * N_shot
            data_support, data_query = data[:p], data[p:]

            proto = model(data_support) # -> size: (way * shot, 1600)
            proto = proto.reshape(N_shot, N_way, -1).mean(dim=0) # -> size: (way, 1600)

            label = torch.arange(N_way).repeat(N_query).long().to(device)

            logits = Similarity.euclidean_distance(model(data_query), proto)
            valid_acc = count_acc(logits, label)

            total_loss += loss.item()
            episodic_acc.append(valid_acc)

        average_loss = total_loss/n_batch_val
        episodic_acc = np.array(episodic_acc)
        acc_mean = episodic_acc.mean()
        acc_std = episodic_acc.std()





