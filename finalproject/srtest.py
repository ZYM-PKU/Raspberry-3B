import os
import cv2
import time
import glob
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from srnet import SRCNN
from PIL import Image, ImageFile
from function import RecurrentSampler
from alive_progress import alive_bar
from torchvision.utils import save_image


PATH = os.path.dirname(__file__)


def Transform():
    '''图像缩放剪切转换'''
    transform_list = [

        #transforms.Resize(size=(512, 512)),
        #transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


tt=transforms.ToTensor()
up2=nn.Upsample(scale_factor=2, mode='bilinear')
down2=nn.MaxPool2d(2)
srcnn=SRCNN(test=True).eval().cuda()
srcnn.load_state_dict(torch.load(os.path.join(PATH,"model/srnet_x2.pth")))




if __name__ == "__main__":
    image = Image.open(os.path.join(PATH,'1.jpg')).convert('RGB')
    image = tt(image).unsqueeze(0).cuda()
    image=up2(image)
    save_image(image.cpu(),os.path.join(PATH,"up2.jpg")) 
    image = srcnn(image)
    save_image(image.cpu(),os.path.join(PATH,"recovered.jpg")) 

