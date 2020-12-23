
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




PATH = os.path.dirname(__file__)


cap = cv2.VideoCapture(0)
cap.set(3,800)
cap.set(4,640)



model_path=os.path.join(PATH,"model/decoder.pth")
encoder=vgg.eval().cuda()
encoder.load_state_dict(torch.load(os.path.join(PATH,'model/vgg_normalised.pth'),map_location="cpu"))
decoder=testdecoder.eval().cuda()
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

tt=transforms.ToTensor()
stylepath=os.path.join(PATH,"3.jpg")
styleimg = Image.open(str(stylepath)).convert('RGB')
styleimg=tt(styleimg).unsqueeze(0).cuda()

sfeature=[]
layers=[31,4,11,18,31]
for i in range(31):
    styleimg = encoder[i](styleimg)
    if i==3 or i==10 or i==17 or i==30:sfeature.append(styleimg)

fbnet=FPnet(encoder,decoder,sfeature).eval().cuda()



while(True):
    tic=time.time()
    ret,frame = cap.read()
    imarray = frame
    cv2.imshow("Origin", imarray)
    image = Image.fromarray(imarray).convert('RGB')
    image = tt(image).unsqueeze(0).cuda()
    output=fbnet(image,alpha=1.0,lamda=1.0,require_loss=False).cpu().squeeze(0).numpy().transpose(1,2,0)
    output=cv2.cvtColor(output,cv2.COLOR_RGB2BGR)
    #save_image(output.cpu(),os.path.join(PATH,"result.jpg"))

    #transfered=np.transpose(transfered[0],(1,2,0))
    #print(transfered.shape)

    #print("1 transfered")
    #cv2.resizeWindow("Transfered",800,640)
    cv2.imshow("Transfered",output) # 显示图像
    key = cv2.waitKey(1) & 0xFF # 等待按键

    toc=time.time()
    print(f"{1.0/(toc-tic)}fps")

    if key == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        break