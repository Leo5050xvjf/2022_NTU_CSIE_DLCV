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


"""
Architecture guidelines for stable Deep Convolutional GANs:
1. Replace any "pooling layers" with "stridden convolutions" (D) and "de-convolutions" (G).
2. Use "batchnorm" in both the G and D.
3. Remove "FC hidden layer" for deeper architectures.
4. Use ReLU activation in the generator for all layers except for the output, which uses Tanh.
5. Use LeakyReLU activation in the discriminator for all layers.

穩定的深度卷積GAN的架構指南：
1.用“跨卷積”（D）和“反捲積”（G）替換任何“池化層”。
2.在 G 和 D 中都使用“ batchnorm”。
3.刪除“ FC隱藏層”以獲得更深的體系結構。
4.除了使用Tanh的輸出外，在G中的所有層上都使用ReLU激活。
5.對所有層在鑑別器中使用LeakyReLU激活。
"""

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 輸出0~1的真實度
        self.outReality = nn.Sequential(
            # in 512 x 4x 4
            nn.Linear(512*1*1,1),
            nn.Sigmoid(),
        )

        # 輸出0~1的機率密度分佈，大小為10x1 ，因為共10種類別而已
        self.outLabels = nn.Sequential(
            # in 512 x 4x 4
            nn.Linear(512 *1*1, 10),
            nn.Sigmoid(),
        )



        self.cnn_disc = nn.Sequential(
            # in: 3 x 28 x 28

            nn.Conv2d(3, 64, kernel_size=(4,4), stride=(2,2), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),
            # out: 64 x 14 x 14

            nn.Conv2d(64, 128, kernel_size=(4,4), stride=(2,2), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),

            # out: 128 x 7 x 7

            nn.Conv2d(128, 256, kernel_size=(4,4), stride=(2,2), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),
            # out: 256 x 3 x 3

            nn.Conv2d(256, 512, kernel_size=(4,4), stride=(2,2), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),
            # out: 512 x 1 x 1

            nn.Flatten(),

            # nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # out: 1 x 1 x 1


        )

    def forward(self, x):

        x = self.cnn_disc(x)
        '''we need to moditfy here to change the output type
        GAN : just output reality 
        ACGAN: 1.output reality  2. output Labels
         '''
        reality = self.outReality(x)
        labels = self.outLabels(x)
        # disc輸出真實度1x1 tensor 還有類別標註 10 x 1 tensor
        return reality,labels


class Generator(nn.Module):
    def __init__(self, z_dim=512):
        super(Generator, self).__init__()
        # embedding 類似一個Linear layer，第1個參數為類別總數(在此為10，數字0-9)，第二個參數為輸出latent dim，
        # 概念:希望自己的labels input對應到多少維度，若輸入為[0,1,0,0,0,0,0,0,0,0] 則希望他的輸出為多少?此數字為任意值的超參數，在此設為512

        self.label_emb = nn.Embedding(10, z_dim)

        self.cnn_gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, kernel_size=(4,4), stride=(1,1), padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # out: 512 x 4 x 4

            nn.ConvTranspose2d(512, 256, kernel_size=(4,4), stride=(2,2), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # out: 256 x 8 x 8

            nn.ConvTranspose2d(256, 128, kernel_size=(4,4), stride=(2,2), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # out: 128 x 16 x 16

            nn.ConvTranspose2d(128, 64, kernel_size=(4,4), stride=(2,2), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # out: 64 x 32 x 32
            nn.Conv2d(64,3,kernel_size = (5,5),stride = (1,1),padding = 0,bias=False),
            # out: 3 x 28 x 28
            nn.Tanh()

        )

    def forward(self, noise,labels):
        # torch.mul的意義在於，由於原本的labels為一個稀疏矩陣(很多0，很少1)，而此種矩陣在計算上浪費空間以及運算量，例如一個1萬維的labels，
        # 則會有將近1萬個0來表達一筆資料，因此embedding則是將這件事情簡化，使計算效率提升，例如[1,0,0]的label輸入，則只會和某幾個weight所內積，而此組weight的位置
        # 也會對應，第0個位置，因此labels與embedding相乘之後，會使每個noise單純去尋找對應的weight相乘，舉例來說，某noise的label為3(one-hot)，則此noise在輸入時，
        # 會自動去尋找label 3 會對應到的weight，進而做內積和GD，而原本的做法則是要先乘上一個稀疏矩陣，[0,0,0,1,0,0..,0]才能找到此noise應該和那些weight相乘，
        # 現在使用embedding後，只要做torch.mul(embedding layer * label)即可代表相同意思。
        # note: labels為 [1,5,2,3,1]的話，則是說 第0個noise -> label = 1 ,第1個noise-> label = 5 接著做的事情都一樣，這樣可以快速的查找每個有label 的noise對應的weight在哪裡，
        # 加速運算速度
        gen_input = torch.mul(self.label_emb(labels),noise)
        gen_input = gen_input.view(gen_input.size(0),gen_input.size(1),1,1)
        img = self.cnn_gen(gen_input)
        return img



if __name__ == "__main__":


    # torch.manual_seed(0)
    # fixed_noise = torch.from_numpy(np.random.normal(0, 1, (3, 512))).to(device="cpu", dtype=torch.float)
    # fixed_noise_labels = torch.tensor([num for _ in range(1) for num in range(3)])
    # G = Generator()
    # img = G(fixed_noise,fixed_noise_labels)
    # D = Discriminator()

    torch.manual_seed(0)
    fixed_noise = torch.from_numpy(np.random.normal(0, 1, (10, 512))).to(device="cpu", dtype=torch.float)
    fixed_noise_labels = torch.tensor([num for _ in range(1) for num in range(10)]).to("cpu")
    gen = Generator()
    disc = Discriminator()
    fake_img_test = gen(fixed_noise, fixed_noise_labels)
    reality_test, img_labels_test = disc(fake_img_test)
    _, labels_test = torch.max(img_labels_test, dim=1)

    for _ in range(10):
        img = fake_img_test[_].view(3,28,28).permute(1,2,0).detach().numpy()
        img = (img-np.min(img)) / (np.max(img)- np.min(img)) * 255
        img = img.astype(np.uint8)
        print(img.shape)
        img = Image.fromarray(img, 'RGB')
        # img.save('./my.png')
        img.show()
        # plt.imshow(img.permute(1, 2, 0))
        # print(img.size())

    plt.imshow(fake_img_test.permute(1, 2, 0))
    print(fake_img_test.size())
    input()
    img_grid_fake = torchvision.utils.make_grid(fake_img_test, nrow=10)
    print(img_grid_fake.size())
    input()

    plt.figure(figsize=[20, 20])
    plt.imshow(img_grid_fake)
    input()
    img_grid_fake = img_grid_fake.float()
    torchvision.utils.save_image(img_grid_fake, f"./fake_test.png")








