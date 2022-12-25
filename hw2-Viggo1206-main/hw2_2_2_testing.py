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
import numpy as np
from models.MyACGAN import Generator
import os
import argparse




def save_tensors_imgs(tensors , label , root = "./acgan_image/"):
    length = len(tensors)
    for _ in range(length):
        img = tensors[_]
        path = os.path.join(root,f"{label}_{_:03d}.png")
        torchvision.utils.save_image(img,path)
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', default="./output_test/hw2_1/testing_50", help='output to test data directory', type=str)
    args = parser.parse_args()
    testing_path = args.test

    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(42)
    fixed_noise = torch.from_numpy(np.random.normal(0, 1, (1000, 50))).to(device=device, dtype=torch.float)

    generator = Generator(50).to(device)
    pth_path = "./checkpoint/ACGAN.pth"
    checkpointFile = torch.load(pth_path, map_location=device)
    generator.load_state_dict(checkpointFile)
    #generator.load_state_dict(torch.load("./checkpoint/ACGAN.pth"),map_location=torch.device(device))
    generator.eval()

    # "./acgan_image/"
    for _ in range(10):
        fixed_noise_labels = torch.tensor([_ for i in range(100)]).to(device=device, dtype=torch.float).to(torch.int64)
        fixed_noise_slice = fixed_noise[_ * 100:(_ + 1) * 100]
        imgs = generator(fixed_noise_slice, fixed_noise_labels)
        save_tensors_imgs(imgs , _ , testing_path)



