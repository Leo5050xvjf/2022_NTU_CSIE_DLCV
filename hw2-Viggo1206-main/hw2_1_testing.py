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
from models.MyinitGAN import Generator


#     .sh file : bash ./hw2_p1.sh $1 ，where $1 is the path we need to generate images
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    z_dim = 1024

    # for i in range(10):
    parser = argparse.ArgumentParser()
    # 0:沒過。1 is :2.07 ，2沒過 。 3:no。4:2.000。5.1.999。6:no。7:2.02。8:1.96。9:2.07
    # parser.add_argument('-a','--randomseed',default="1",help = "chose a fixed randomseed",type = str)
    parser.add_argument('-o','--output',default="./",help = "chose a fixed randomseed",type = str)

    fixed_seed = 1
    arg = parser.parse_args()
    output_path = arg.output
    torch.manual_seed(fixed_seed)


    with torch.no_grad():
        gen = Generator(z_dim).to(device)
        gen.load_state_dict(torch.load(f"./checkpoint/GAN175.pth"))
        gen.eval()
        fixed_noise = torch.randn(1000, z_dim, 1, 1).to(device)
        fixed_noise_32 = fixed_noise[:32]
        for _ in range(1000):
            noise = fixed_noise[_].view(1,z_dim,1,1)
            img = gen(noise)
            img = (img+1)/2
            outpath =os.path.join(output_path,f"{_:03d}.png")
            torchvision.utils.save_image(img,outpath )

