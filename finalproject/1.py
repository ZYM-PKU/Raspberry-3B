from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import os
import cv2 # 导入需要的库
import torch
import numpy as np
from convert import transfer
from PIL import Image, ImageFile
from net import vgg,FPnet,Decoder,testdecoder
import torchvision.transforms as transforms
from torchvision.utils import save_image


resolution=(64,32)
camera = PiCamera()
camera.resolution = resolution # 设置分辨率
camera.framerate = 90 # 设置帧率
rawCapture = PiRGBArray(camera, size=resolution)
time.sleep(0.1) # 等待摄像头模块初始化

PATH = os.path.dirname(__file__)




model_path=os.path.join(PATH,"model/decoder.pth")
encoder=vgg.eval()
encoder.load_state_dict(torch.load(os.path.join(PATH,'model/vgg_normalised.pth'),map_location="cpu"))
decoder=testdecoder.eval()
decoder.load_state_dict(torch.load(os.path.join(PATH,"model/decoder.pth"),map_location="cpu"))
for param in encoder.parameters():
    param.requires_grad = False
for param in decoder.parameters():
    param.requires_grad = False



def test_transform(size):
    transform_list = []
    transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

mytransfer=test_transform(64)
stylepath=os.path.join(PATH,"2.jpg")
styleimg = Image.open(str(stylepath)).convert('RGB')
styleimg=mytransfer(styleimg).unsqueeze(0)

sfeature=[]
layers=[31,4,11,18,31]
for i in range(31):
    styleimg = encoder[i](styleimg)
    if i==3 or i==10 or i==17 or i==30:sfeature.append(styleimg)

fbnet=FPnet(encoder,decoder,sfeature).eval()



tt=transforms.ToTensor()

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    tic=time.time()
    imarray = frame.array
    image = Image.fromarray(imarray).convert('RGB')
    image = tt(image).unsqueeze(0)
    output=fbnet(image,alpha=1.0,lamda=1.0,require_loss=False)
    #output=cv2.cvtColor(output,cv2.COLOR_RGB2BGR)
    save_image(output.cpu(),os.path.join(PATH,"result.jpg"))

    #transfered=np.transpose(transfered[0],(1,2,0))
    #print(transfered.shape)

    print("1 transfered")
    #cv2.resizeWindow("Transfered",800,640)
    cv2.imshow("Transfered", cv2.imread(os.path.join(PATH,"result.jpg"))) # 显示图像
    key = cv2.waitKey(1) & 0xFF # 等待按键
    rawCapture.truncate(0) # 准备下一副图像
    toc=time.time()
    print(f"{1.0/(toc-tic)}fps")

    if key == ord("q"):
        break