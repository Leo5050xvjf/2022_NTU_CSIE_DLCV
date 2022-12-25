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

from models.FCN32 import MyFCN32

import glob
import time
import datetime


cls_color = {
    0:  [0, 255, 255],
    1:  [255, 255, 0],
    2:  [255, 0, 255],
    3:  [0, 255, 0],
    4:  [0, 0, 255],
    5:  [255, 255, 255],
    6: [0, 0, 0],
}


class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.img_names = [(os.path.join(self.root, f)) for f in os.listdir(self.root) if "sat" in f]
        self.mask_names = [sat.replace("sat", "mask").replace("jpg", "png") for sat in self.img_names]
        self.file_nums = [f.replace("_sat.jpg", "") for f in os.listdir(self.root) if "sat" in f]
        self.length = len(self.img_names)

    def __getitem__(self, index):
        img_name, mask_name = self.img_names[index], self.mask_names[index]
        file_num = self.file_nums[index]

        img = Image.open(img_name)

        mask = np.array(Image.open(mask_name), dtype="uint8")
        mask = self.make_mask(mask)

        if self.transform != None:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask, file_num

    def __len__(self):
        return self.length

    def make_mask(self, mask: np):
        result = np.empty(mask.shape[:2], dtype=np.int32)
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        result[mask == 3] = 0  # (Cyan: 011) Urban land
        result[mask == 6] = 1  # (Yellow: 110) Agriculture land
        result[mask == 5] = 2  # (Purple: 101) Rangeland
        result[mask == 2] = 3  # (Green: 010) Forest land
        result[mask == 1] = 4  # (Blue: 001) Water
        result[mask == 7] = 5  # (White: 111) Barren land
        result[mask == 0] = 6  # (Black: 000) Unknown
        result[mask == 4] = 6  # (Red: 100) Unknown
        return result


def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target.long(), weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss


def make_mask(mask):
    h, w = mask.shape[:2]
    result = np.zeros((h, w, 3), dtype='uint8')
    for i in range(7):
        result[mask == i] = np.array(cls_color[i], dtype='uint8')

    return result

def train(model, train_loader, optimizer, epoch, log_interval=100, start_time="", device="cuda"):
    iteration = 0
    correct = 0
    train_loss = 0
    for batch_idx, (data, labels, file_num) in enumerate(train_loader):  # for 每個 mini batch
        N = data.size(0)
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = cross_entropy2d(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.data.item()

        mask = output[0].cpu().detach().numpy()
        mask = np.argmax(mask, axis=0)
        mask = make_mask(mask)
        Image.fromarray(mask).save(f"./fcn32/{start_time}/train/{ep}/{file_num[0]}.png")

        if iteration % log_interval == 0 and iteration != 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.0f}%)'.format(
                ep, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(),
                correct, batch_idx * len(data), 100. * correct / (batch_idx * len(data))))
        iteration += 1

    return train_loss/len(train_loader.dataset)


def test(model, testset_loader, ep, start_time, device="cuda"):
    criterion = cross_entropy2d
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for batch_idx, (data, labels, file_num) in enumerate(testset_loader):
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            test_loss += criterion(output, labels).data.item() # sum up batch loss

            mask = output[0].cpu().detach().numpy()
            mask = np.argmax(mask, axis=0)
            mask = make_mask(mask)
            Image.fromarray(mask).save(f"./fcn32/{start_time}/valid/{ep}/{file_num[0]}.png")



    test_loss /= len(testset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testset_loader.dataset),
        100. * correct / len(testset_loader.dataset)))

    return test_loss

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)



if __name__ == '__main__':

    epoch = 40
    device = "cuda"

    torch.manual_seed(1)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    t = transforms.ToTensor()
    train_set = MyDataset(root=r".\p2_data\train", transform=t)
    val_set = MyDataset(root=r".\p2_data\validation", transform=t)
    print('# images in train_set:', len(train_set))  # Should print 60000
    print('# images in val_set:', len(val_set))  # Should print 10000

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

    model = MyFCN32().to(device)

    lr = 0.001
    optimizer = optim.SGD([
        {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * lr},
        {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'],
         'lr': lr, 'weight_decay': 1e-4}
    ], momentum=0.9)

    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    t1 = time.time()
    train_loss_list = []
    test_loss_list = []

    print(start_time)
    try:
        os.makedirs(f'./fcn32/{start_time}')
        os.makedirs(f'./fcn32/{start_time}/train')
        os.makedirs(f'./fcn32/{start_time}/valid')
    except Exception as e:
        print(e)

    for ep in range(epoch):
        te1 = time.time()
        try:
            os.makedirs(f'./fcn32/{start_time}/train/{ep}')
            os.makedirs(f'./fcn32/{start_time}/valid/{ep}')
        except Exception as e:
            print(e)

        train_loss = train(model, train_loader, optimizer, ep, 100, start_time, device=device)
        save_checkpoint(f"./checkpoint/{start_time}-fcn32-{ep}.pth", model, optimizer)
        test_loss = test(model, val_loader, ep, start_time, device=device)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        te2 = time.time()
        print(f"epoch {ep} cost {te2 - te1} sec")


    t2 = time.time()
    print(f"cost: {t2 - t1} sec")



