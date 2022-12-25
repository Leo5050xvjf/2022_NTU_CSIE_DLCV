import torch

from transformers import BertTokenizer
from PIL import Image
import argparse

from models import caption
from datasets import coco, utils
from configuration import Config
import os
import torchvision.transforms.functional as F
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as t
import numpy as np
import parser
import os

config = Config()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('-inputpath', type=str, help='input path to image',default= "../hw3_data/p2_data/images")
    parser.add_argument('-outputpath', type=str, help='output path to image', default='./')
    args = parser.parse_args()
    image_file = args.inputpath
    image_output = args.outputpath

    img_names = os.listdir(image_file)
    img_paths = [os.path.join(image_file,img_name) for img_name in img_names]
    # checkpoint_path = "./checkpoint_p2.pth"



    checkpoint_path = "./CPTR_Pretrain.pth"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model,_ = caption.build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint)



    start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
    end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
    image_path = "../hw3_data/p2_data/images/bike.jpg"
    def generate_five_img(image_path,image_output_path):

        image = Image.open(image_path)
        img_show = image
        image = coco.val_transform(image)
        image = image.unsqueeze(0)


        def create_caption_and_mask(start_token, max_length):
            caption_template = torch.zeros((1, max_length), dtype=torch.long)
            mask_template = torch.ones((1, max_length), dtype=torch.bool)

            caption_template[:, 0] = start_token
            mask_template[:, 0] = False

            return caption_template, mask_template

        caption, cap_mask = create_caption_and_mask(
            start_token, config.max_position_embeddings)
        @torch.no_grad()
        def evaluate():
            model.eval()
            att_bag = []
            pos_bag = []
            for i in range(config.max_position_embeddings - 1):

                predictions,att,pos = model(image, caption, cap_mask)
                att_bag.append(att)
                pos_bag.append(pos)
                predictions = predictions[:, i, :]
                predicted_id = torch.argmax(predictions, axis=-1)

                if predicted_id[0] == 102:
                    return caption,att_bag,pos_bag

                caption[:, i+1] = predicted_id[0]
                cap_mask[:, i+1] = False
            return caption,att_bag,pos_bag

        output,att_bag,pos_bag = evaluate()
        result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        words  = tokenizer.tokenize(result)
        # print(test)
        # input()

        # words = result.split(" ")
        # last_word = words.pop(-1).split(".")
        # words.append(last_word[0])
        # words.append(".")
        num_image = len(words)

        pos_size = pos_bag[0][0].size()[2:]
        resize_h,resize_w = pos_size[0],pos_size[1]

        def show_att_map(att_map,img_show,resize_h,resize_w):
            fig = plt.figure(figsize=(16, 8))

            ax = fig.add_subplot(2, num_image//2 + 1, 1)
            plt.title("<start>")
            plt.xticks([])
            plt.yticks([])
            img_show = img_show.resize((200,200))
            plt.imshow(img_show)
            att_matrix = att_map[-1]
            att_matrix = torch.squeeze(att_matrix)

            for i,(word,map) in enumerate(zip(words,att_matrix)):

                map = map.numpy()
                map = map.reshape(resize_h,resize_w)
                map = cv2.resize(map,(200,200))
                map_merge = np.zeros((200,200,3))

                map_merge[:,:,0] = map
                map_merge[:,:,1] = map
                map_merge[:,:,2] = map
                map_merge = Image.fromarray(np.uint8(map*255))
                img_show = img_show.convert('L')
                map_merge = Image.blend(img_show,map_merge,0.99)

                ax = fig.add_subplot(2, num_image//2 + 1, i+2)
                plt.title(f"{word}")
                plt.xticks([])
                plt.yticks([])
                plt.imshow(map_merge)
                plt.savefig(f"{image_output_path}")
            # plt.show()
        show_att_map(att_bag,img_show,resize_h,resize_w)
    out_path = "./"


    for img_path,img_name in zip(img_paths,img_names):

        img_name = img_name.split(".")[0]+".png"
        output_img_path = os.path.join(image_output,img_name)
        generate_five_img(img_path,output_img_path)
