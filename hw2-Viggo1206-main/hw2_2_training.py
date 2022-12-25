import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from models.MyACGAN import Discriminator, Generator
from torch.utils.tensorboard import SummaryWriter
from datasets.MyDataset import DigitsDataset

note = '''

=====================================================
'''



if __name__ == '__main__':

    '''*****************Hyperparameters*********************'''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    img_size = 28
    batch_size = 128
    epochs = 200
    n_mean = 0
    n_std = 0.02
    lr = 2e-4
    betas = (0.5, 0.999)
    filter_num = 8
    z_dim = 50
    '''*****************Hyperparameters*********************'''


    disc = Discriminator().to(device)
    gen = Generator(z_dim).to(device)


    # 目前測試0~9數字各10張 共100張，固定seed的label
    # fixed_noise = torch.randn(batch_size, z_dim, 1, 1).to(device)

    # print(fixed_noise_labels)
    # input()

    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = DigitsDataset("../HW2/yuchin/hw2_data/digits/mnistm/train","../HW2/yuchin/hw2_data/digits/mnistm/train.csv", t)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,)

    opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=betas)
    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=betas)

    step = 0

    adversarial_loss = torch.nn.BCELoss().to(device)
    auxiliary_loss = torch.nn.CrossEntropyLoss().to(device)
    torch.manual_seed(0)
    wanna_show_num = 100
    fixed_noise = torch.from_numpy(np.random.normal(0, 1, (wanna_show_num, z_dim))).to(device=device, dtype=torch.float)
    fixed_noise_labels = torch.tensor([num for _ in range(10) for num in range(10)]).to(device)


    for ep in range(1, epochs + 1):
        print(f"ep = {ep}")
        disc.train()
        gen.train()
        for batch_idx, (img, label,file_name) in enumerate(train_loader):


            """**************************Train Generatator******************************"""
            batch_size_num = len(label)
            # Generate fake images
            # fake data and fake data labels
            opt_gen.zero_grad()
            # noise SIZE : batch_size x 512
            noise = torch.from_numpy(np.random.normal(0, 1, (batch_size_num, z_dim))).to(device=device, dtype=torch.float).to(torch.int64)
            # noise_label SIZE : 1 x batch_size
            noise_label = np.random.randint(0, 10, batch_size_num)

            # noise_label_matrix事宜個稀疏矩陣，[[0,1,0,0],[1,0,0,0]...]用來與predictd的做Cross Entrophy
            noise_label =torch.from_numpy(noise_label).to(device=device, dtype=torch.float).to(torch.int64)

            gen_img = gen(noise,noise_label)
            gen_img_reality,gen_img_pred = disc(gen_img)

            real_reality = Variable(torch.FloatTensor(batch_size_num, 1).fill_(1.0), requires_grad=False).to(device)
            # (real_reality,gen_img_reality) -> 希望gen 能產生的reality變高，因此real_reality 是1
            # 真實度loss

            gen_reality_loss = adversarial_loss(gen_img_reality,real_reality)
            noise_label = noise_label.type(torch.LongTensor).to(device)
            gen_label_loss = auxiliary_loss(gen_img_pred,noise_label.view(-1) )
            # 更新generator loss
            gen_loss = (gen_reality_loss+gen_label_loss)*0.5
            gen_loss.backward()
            opt_gen.step()

            """**************************Train Generatator******************************"""

            """**************************Train Discriminator******************************"""

            img = img.to(device)
            img_label = label.to(device)
            # Train Discriminator: max log(D(img)) + log(1 - D(G(z))
            opt_disc.zero_grad()
            # Pass img images through discriminator
            # img_reality batch_size x 1x1 ,img_labels 10 x 1
            img_reality,img_pred = disc(img)
            # disc 要分辨出真實影像，因此真實影像的reality是1
            img_reality_loss = adversarial_loss(img_reality ,real_reality)
            # 真實影像有正確labels
            img_labels_loss = auxiliary_loss(img_pred,img_label.view(-1))
            # 計算真實影像產生的loss
            d_real_loss = (img_reality_loss+img_labels_loss) *0.5

            # Loss for fake image
            fake_reality,fake_labels = disc(gen_img.detach())
            # 生成影像應該要被分辨出來，所以真實度是0，
            fake = Variable(torch.FloatTensor(batch_size_num, 1).fill_(0.0), requires_grad=False).to(device)
            # 假影像的reality是0
            d_fake_BCE_loss = adversarial_loss(fake_reality, fake)
            d_fake_CE_loss = auxiliary_loss(fake_labels,noise_label.view(-1))
            d_fake_loss =( d_fake_BCE_loss+d_fake_CE_loss )*0.5
            # 真、假影像產生的Loss加總
            d_total_loss = (d_real_loss+ d_fake_loss) * 0.5
            d_total_loss.backward()
            opt_disc.step()

            """**************************Train Discriminator******************************"""
        with torch.no_grad():
            # print("with no grad()!")
            disc.eval()
            gen.eval()
            fake_img_test = gen(fixed_noise,fixed_noise_labels)
            reality_test,img_labels_test = disc(fake_img_test)
            _,labels_test = torch.max(img_labels_test,dim=1)

            print("accuracy : ",sum(labels_test == fixed_noise_labels) / len(fixed_noise_labels))
            print(f"fixed_noise_labels len is : {len(fixed_noise_labels)}")
            # print(fake_img_test.size())
            # input()

            img_grid_fake = torchvision.utils.make_grid(fake_img_test,nrow=10,normalize=True,scale_each=True)
            img_grid_fake = img_grid_fake.float()
            torchvision.utils.save_image(img_grid_fake, f"../HW2/yuchin/acgan_image/dim50/fake_{ep}.png",)

        if ep % 5 == 0:
            # 目前85.pth 有90%正確率
            torch.save(gen.state_dict(),f"../HW2/yuchin/acgan_checkpoint/dim50/{ep}.pth")














