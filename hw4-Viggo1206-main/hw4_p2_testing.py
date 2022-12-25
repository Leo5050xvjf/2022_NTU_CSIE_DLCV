import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet50
import pandas as pd
from PIL import Image



class OfficeHomeDataset(Dataset):
    def __init__(self, data_dir, csv_path):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")
        self.filenames = self.data_df["filename"].tolist()
        self.labels = self.data_df["label"].tolist()
        self.strToNum = {'Couch': 0, 'Helmet': 1, 'Refrigerator': 2, 'Alarm_Clock': 3, 'Bike': 4, 'Bottle': 5, 'Calculator': 6, 'Chair': 7, 'Mouse': 8, 'Monitor': 9, 'Table': 10, 'Pen': 11, 'Pencil': 12, 'Flowers': 13, 'Shelf': 14, 'Laptop': 15, 'Speaker': 16, 'Sneakers': 17, 'Printer': 18, 'Calendar': 19, 'Bed': 20, 'Knives': 21, 'Backpack': 22, 'Paper_Clip': 23, 'Candles': 24, 'Soda': 25, 'Clipboards': 26, 'Fork': 27, 'Exit_Sign': 28, 'Lamp_Shade': 29, 'Trash_Can': 30, 'Computer': 31, 'Scissors': 32, 'Webcam': 33, 'Sink': 34, 'Postit_Notes': 35, 'Glasses': 36, 'File_Cabinet': 37, 'Radio': 38, 'Bucket': 39, 'Drill': 40, 'Desk_Lamp': 41, 'Toys': 42, 'Keyboard': 43, 'Notebook': 44, 'Ruler': 45, 'ToothBrush': 46, 'Mop': 47, 'Flipflops': 48, 'Oven': 49, 'TV': 50, 'Eraser': 51, 'Telephone': 52, 'Kettle': 53, 'Curtains': 54, 'Mug': 55, 'Fan': 56, 'Push_Pin': 57, 'Batteries': 58, 'Pan': 59, 'Marker': 60, 'Spoon': 61, 'Screwdriver': 62, 'Hammer': 63, 'Folder': 64}
        self.numToStr = {0: 'Couch', 1: 'Helmet', 2: 'Refrigerator', 3: 'Alarm_Clock', 4: 'Bike', 5: 'Bottle', 6: 'Calculator', 7: 'Chair', 8: 'Mouse', 9: 'Monitor', 10: 'Table', 11: 'Pen', 12: 'Pencil', 13: 'Flowers', 14: 'Shelf', 15: 'Laptop', 16: 'Speaker', 17: 'Sneakers', 18: 'Printer', 19: 'Calendar', 20: 'Bed', 21: 'Knives', 22: 'Backpack', 23: 'Paper_Clip', 24: 'Candles', 25: 'Soda', 26: 'Clipboards', 27: 'Fork', 28: 'Exit_Sign', 29: 'Lamp_Shade', 30: 'Trash_Can', 31: 'Computer', 32: 'Scissors', 33: 'Webcam', 34: 'Sink', 35: 'Postit_Notes', 36: 'Glasses', 37: 'File_Cabinet', 38: 'Radio', 39: 'Bucket', 40: 'Drill', 41: 'Desk_Lamp', 42: 'Toys', 43: 'Keyboard', 44: 'Notebook', 45: 'Ruler', 46: 'ToothBrush', 47: 'Mop', 48: 'Flipflops', 49: 'Oven', 50: 'TV', 51: 'Eraser', 52: 'Telephone', 53: 'Kettle', 54: 'Curtains', 55: 'Mug', 56: 'Fan', 57: 'Push_Pin', 58: 'Batteries', 59: 'Pan', 60: 'Marker', 61: 'Spoon', 62: 'Screwdriver', 63: 'Hammer', 64: 'Folder'}
        self.ToTA = 1


        self.transform = transforms.Compose([
            transforms.Resize([128,128]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(os.path.join(self.data_dir, filename))
        image = self.transform(img)
        strLabel = self.labels[index]
        numLabel = self.strToNum[strLabel]

        # 在測試的時候，只讀的到圖片，還有檔名所以只要return這兩個就好
        return index,image, filename

    def __len__(self):
        return len(self.data_df)


if __name__ == '__main__':


    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = resnet50(pretrained=False)
    net.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(2048, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 65),
            )

    # pth選擇，本次檔案較小，把路徑寫死即可，不需要丟dropbox
    checkpoint = torch.load("./60.pth", map_location=device)
    net.load_state_dict(checkpoint)
    net.to(device)

    parser = argparse.ArgumentParser(description="BYOL")
    parser.add_argument('--testCSV', default="./hw4_data/office/val.csv", type=str, help='testing images csv file')
    parser.add_argument('--testImage', default="./hw4_data/office/val", type=str, help='testing images directory')
    parser.add_argument('--output', default="./pred.csv", type=str, help='path of output csv file')
    arg = parser.parse_args()
    imagePATH = arg.testImage
    testCSV = arg.testCSV
    outpath = arg.output



    valDataset = OfficeHomeDataset(imagePATH ,testCSV)
    valDataLoader = DataLoader(dataset=valDataset,batch_size=1,shuffle=False)

    df = {"id":[],
          "filename":[],
          "label":[]}

    with torch.no_grad():
        net.eval()
        numTostr = valDataset.numToStr
        # print(len(valDataLoader))
        for ID,image,filename in valDataLoader:
            image = image.to(device)
            pred = net(image)
            _,label = torch.max(pred,dim=1)
            label = label.to("cpu")
            label = numTostr[label.item()]
            df["id"].append(int(ID))
            df["filename"].append(filename[0])
            df["label"].append(label)
    df = pd.DataFrame(df)
    df.to_csv(outpath,index = False)

    # 測試輸出正確率

    # ans = pd.read_csv("/home/yiting/Documents/DLCV/hw4/hw4_data/office/val.csv").set_index("id")
    # ans = ans["label"].tolist()
    # pred_ = pd.read_csv("./p2_pred.csv").set_index("id")
    # pred_ = pred_["label"].tolist()
    # num =len(pred_)
    # counter = 0
    # for pred_ans,label in zip(pred_,ans):
    #     if pred_ans == label:
    #         counter+=1
    # print(f"the acc is {(counter/num) * 100}%")




