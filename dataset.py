import torch
import torchvision.transforms as transforms
import numpy as np
import os
import PIL.Image as image
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt

data_dir = r'F:\ycq\UNET\p_data'
label_dir = r'F:\ycq\UNET\p_label'

transform = transforms.Compose([
    transforms.ToTensor()
])
class Get_data(Dataset):
    def __init__(self):
        self.dataset = os.listdir(data_dir)
        self.dataset.sort(key=lambda x: int(x.split('.')[0]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img = image.open(os.path.join(data_dir,self.dataset[index]))
        # imgs = tosqure(img,572)
        img = img.resize((572,572),image.ANTIALIAS)
        label = image.open(os.path.join(label_dir,self.dataset[index]))
        # label =label.resize(388,388)
        # labels = tosqure(label,388,flg=True)
        label = label.resize((388,388),image.ANTIALIAS)
        array_img = transform(img)
        array_label = transform(label)
        return array_img,array_label

def tosqure(img,size,flg=False):
    if flg:
        w,h =img.size
        img = img.resize((size,int(h*size/w)))
        new_img = image.new('RGB', (size, size), (128, 128, 128))
        w,h =img.size
        new_img.paste(img, (0, int((size - h) / 2)))
    else:
        w,h = img.size
        new_img = image.new('RGB',(size,size),(128,128,128))
        new_img.paste(img,(int((size-w)/2),int((size-h)/2)))
    return new_img

# get_data = Get_data()
# get_data.__getitem__(3)
