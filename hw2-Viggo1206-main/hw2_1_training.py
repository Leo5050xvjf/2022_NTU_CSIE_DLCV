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
from datasets.MyDataset import FaceDataset
from models.MyinitGAN import Discriminator, Generator,weights_init


# 檔案路徑更改，整個code 有用到路徑的地方要再檢查一次


if __name__ == '__main__':

    # Hyperparameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 64
    batch_size = 128
    epochs = 500
    n_mean = 0
    n_std = 0.02

    lr = 2e-4
    betas = (0.5, 0.999)
    filter_num = 8
    z_dim_list = [1024]
    num_list = [5]
    for z_dim,num in zip(z_dim_list,num_list):
        disc = Discriminator().to(device)
        gen = Generator(z_dim).to(device)
        fixed_noise = torch.randn(batch_size, z_dim, 1, 1).to(device)

        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_set = FaceDataset("./hw2_data/face/train", t)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, )

        opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=betas)
        opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=betas)

        step = 0
        for ep in range(1, epochs + 1):
            print(f"ep = {ep}")

            for batch_idx, (real, _) in enumerate(train_loader):
                real = real.to(device)

                # Train Discriminator: max log(D(real)) + log(1 - D(G(z))
                opt_disc.zero_grad()

                # Pass real images through discriminator
                real_preds = disc(real).view(-1)
                real_targets = torch.ones_like(real_preds).to(device)
                real_loss = F.binary_cross_entropy(real_preds, real_targets)

                # Generate fake images
                latent = torch.randn(batch_size, z_dim, 1, 1).to(device)
                fake_images = gen(latent)
                fake_preds = disc(fake_images).view(-1)

                fake_targets = torch.zeros_like(fake_preds).to(device)
                fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)

                lossD = real_loss + fake_loss
                lossD.backward()
                opt_disc.step()

                # Train Generator: min log(1 - D(G(z))) -> max D(G(z))
                # Clear generator gradients
                opt_gen.zero_grad()

                # Generate fake images
                latent = torch.randn(batch_size, z_dim, 1, 1).to(device)
                fake_images = gen(latent)

                # Try to fool the discriminator
                preds = disc(fake_images)
                targets = torch.ones(batch_size, 1).to(device)
                lossG = F.binary_cross_entropy(preds, targets)

                # Update generator weights
                lossG.backward()
                opt_gen.step()

            with torch.no_grad():
                fake = gen(fixed_noise)

                img_grid_fake = torchvision.utils.make_grid(fake,nrow=8,normalize=True,scale_each=True)
                img_grid_fake = img_grid_fake.float()

                torchvision.utils.save_image(img_grid_fake, f"./gan_image_grid/{num}/fake_{ep}.png")

            if ep % 5 == 0:
                torch.save(gen.state_dict(),f"./gan_checkpoint/{num}/{ep}.pth")
                # save gen 即可，因為只有他要輸出影像，不是存disc!













