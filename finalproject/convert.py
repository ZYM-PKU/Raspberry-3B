import os
import cv2
import glob
import torch
import argparse
import numpy as np
import torchvision.transforms as transforms
from net import vgg,FPnet,Decoder,testdecoder
from PIL import Image, ImageFile
from function import coral,change_color
from torchvision.utils import save_image


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('-c','--content',type=str, 
                    help='path of your content image')
parser.add_argument('-s','--style',type=str, 
                    help='path of your style image')

args = parser.parse_args()


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#训练设备


def test_transform(size):
    transform_list = []
    transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def transfer(contentimg,fbnet,converted="result.jpg"):#20200521decoder10000_1.pth
    '''一次前传得到风格化图像'''
    #mytransfer=test_transform(pixel)

    #contentimg = Image.open(str(contentpath)).convert('RGB')
    #styleimg = Image.open(str(stylepath)).convert('RGB')

    #contentimg=mytransfer(contentimg).unsqueeze(0)
    #styleimg=mytransfer(styleimg).unsqueeze(0)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #decoder=Decoder().to(device).eval()
    #decoder=testdecoder
    #decoder.load_state_dict(torch.load(model_path,map_location="cpu"))

    # decoder = decoder.module
    # decoder.load_state_dict(torch.load(model_path))


    output=fbnet(contentimg,alpha=1.0,lamda=1.0,require_loss=False)
    
    #save_image(output.cpu(),converted)
    contentimg.detach()
    output.detach()
    return True



if __name__ == "__main__":
    decoder=testdecoder
    decoder.load_state_dict(torch.load('model/decoder.pth',map_location="cpu"))
    encoder=vgg
    encoder.load_state_dict(torch.load('model/vgg_normalised.pth',map_location="cpu"))
    transfer(args.content,args.style,encoder,decoder)



