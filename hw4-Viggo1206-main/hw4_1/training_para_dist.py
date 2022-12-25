import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from yuchin.utils.dataset import MiniDataset
from yuchin.utils.samplers import GeneratorSampler
from yuchin.utils.util import Similarity, count_acc
from yuchin.models.convnet import Convnet

note = '''
V1.

SGD with mo
=====================================================
'''

class Dist_Net(nn.Module):
    def __init__(self, N_way):
        super(Dist_Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * N_way * 1600, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.Linear(64, N_way),
        )
    def forward(self, x):
        x = self.fc(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N-way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N-shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument( '--N-query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--test_csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str, help="Testing images directory")
    parser.add_argument('--testcase_csv', type=str, help="Test case csv")
    parser.add_argument('--output_csv', type=str, help="Output filename")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # tools setting
    import datetime
    import time
    # Hyper parameters
    n_batch = 600
    n_batch_val = 50   # 投影片規定
    lr = 0.001
    epoch = 300
    device = "cuda" if torch.cuda.is_available() else "cpu"



    # Data loader
    train_set = MiniDataset('../hw4_data/mini/train', '../hw4_data/mini/train.csv')
    sampler = GeneratorSampler(n_batch, args.N_way, args.N_shot + args.N_query, cls_num=64)
    train_loader = DataLoader(dataset=train_set, batch_sampler=sampler)

    valid_set = MiniDataset('../hw4_data/mini/val', '../hw4_data/mini/val.csv')
    sampler = GeneratorSampler(n_batch_val, args.N_way, args.N_shot + args.N_query, cls_num=16)
    valid_loader = DataLoader(dataset=valid_set, batch_sampler=sampler)

    # Model & Optimizer

    check_path = "./checkpoints/_maxacc_ep_100_k1.pth"
    model = Convnet().to(device=device)
    checkpoint = torch.load(check_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    model_dist = Dist_Net(args.N_way).to(device)
    optimizer = torch.optim.Adam([{'params': model.parameters()},
                                  {'params': model_dist.parameters(), 'lr': 1e-5}], lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)


    # Training
    for ep in tqdm(range(1, epoch+1)):

        model.train()
        model_dist.train()
        episodic_acc = []
        total_loss = 0

        t1 = time.time()
        for batch_idx, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()

            data = data.to(device)
            p = args.N_way * args.N_shot
            data_support, data_query = data[:p], data[p:]

            proto = model(data_support) # -> size: (way * shot, 1600)
            proto = proto.reshape(args.N_shot, args.N_way, -1).mean(dim=0) # -> size: (way, 1600)

            label = torch.arange(args.N_way).repeat(args.N_query).long().to(device)

            # logits = Similarity.euclidean_distance(model(data_query), proto)
            # print(logits.size())
            proto_query = model(data_query)
            proto_query = proto_query.unsqueeze(1).expand(args.N_query * args.N_way, args.N_way, -1)
            proto = proto.unsqueeze(0).expand(args.N_query * args.N_way, args.N_way, -1)
            proto_query = proto_query.reshape(args.N_query * args.N_way, -1)
            proto = proto.reshape(args.N_query * args.N_way, -1)

            data_cat = torch.cat((proto_query, proto), dim=1)
            logits = model_dist(data_cat)

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
        model_dist.eval()
        episodic_acc = []
        total_loss = 0

        for batch_idx, (data, _) in enumerate(valid_loader):
            data = data.to(device)
            p = args.N_way * args.N_shot
            data_support, data_query = data[:p], data[p:]

            proto = model(data_support) # -> size: (way * shot, 1600)
            proto = proto.reshape(args.N_shot, args.N_way, -1).mean(dim=0) # -> size: (way, 1600)

            label = torch.arange(args.N_way).repeat(args.N_query).long().to(device)

            # logits = Similarity.euclidean_distance(model(data_query), proto)
            # print(logits.size())
            proto_query = model(data_query)
            proto_query = proto_query.unsqueeze(1).expand(args.N_query * args.N_way, args.N_way, -1)
            proto = proto.unsqueeze(0).expand(args.N_query * args.N_way, args.N_way, -1)
            proto_query = proto_query.reshape(args.N_query * args.N_way, -1)
            proto = proto.reshape(args.N_query * args.N_way, -1)

            data_cat = torch.cat((proto_query, proto), dim=1)
            logits = model_dist(data_cat)

            valid_acc = count_acc(logits, label)


            total_loss += loss.item()
            episodic_acc.append(valid_acc)


        average_loss = total_loss/n_batch_val
        episodic_acc = np.array(episodic_acc)
        acc_mean = episodic_acc.mean()
        print(f"ep is {ep} ,valid_acc is {acc_mean}")
        acc_std = episodic_acc.std()





