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
from srnet import SRCNN,SRNET
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

srnet=SRNET()
srnet.load_state_dict(torch.load(os.path.join(PATH,"model_epoch_150.pth")))

transform=Transform()
img = Image.open(os.path.join(PATH,"result.jpg")).convert('RGB')
img=transform(img).unsqueeze(0)
img=srnet(img)

save_image(img.cpu(),os.path.join(PATH,"recovered.jpg")) 

