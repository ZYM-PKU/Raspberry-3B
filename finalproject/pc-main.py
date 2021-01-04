import time
import os
import cv2 # 导入需要的库
import torch
import torch.nn as nn
import numpy as np
from convert import transfer
from PIL import Image, ImageFile
from net import vgg,FPnet,Decoder,testdecoder
import torchvision.transforms as transforms
from torchvision.utils import save_image
from srnet import SRCNN


PATH = os.path.dirname(__file__)
SCALE = (400,320)#帧大小


cap = cv2.VideoCapture(0)
cap.set(3,SCALE[0])
cap.set(4,SCALE[1])#w:h=5:4
#cap.set(cv2.CAP_PROP_FPS,60)


#加载模型
encoder=vgg.eval().cuda()
encoder.load_state_dict(torch.load(os.path.join(PATH,'model/vgg_normalised.pth'),map_location="cpu"))
decoder=testdecoder.eval().cuda()
decoder.load_state_dict(torch.load(os.path.join(PATH,"model/decoder.pth"),map_location="cpu"))
for param in encoder.parameters():
    param.requires_grad = False
for param in decoder.parameters():
    param.requires_grad = False

fbnet=FPnet(encoder,decoder).eval().cuda()



#初始化风格特征
def test_transform(size):
    transform_list = []
    transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

tt=transforms.ToTensor()
transform = test_transform(SCALE[1])
up2=nn.Upsample(scale_factor=2, mode='bilinear')
srcnn=SRCNN(test=True).eval().cuda()
srcnn.load_state_dict(torch.load(os.path.join(PATH,"model/srnet_x2.pth")))

def set_style(style_path):
    global fbnet
    #设置风格
    stylepath = os.path.join(PATH,style_path)
    styleimg = Image.open(str(stylepath)).convert('RGB')
    styleimg = tt(styleimg).unsqueeze(0).cuda()

    sfeature=[]
    layers=[31,4,11,18,31]
    for i in range(31):
        styleimg = encoder[i](styleimg)
        if i==3 or i==10 or i==17 or i==30:sfeature.append(styleimg)

    fbnet.set_style_feature(sfeature)



def video_mod():


    while(True):

        tic=time.time()
        ret,frame = cap.read()
        imarray = frame
        cv2.imshow("Origin", imarray)
        image = Image.fromarray(imarray).convert('RGB')
        image = tt(image).unsqueeze(0).cuda()
        output=fbnet(image,alpha=1.0,lamda=1.0,require_loss=False)
        #output=up2(output)
        #cv2.imshow("Transfered_l",cv2.cvtColor(output.cpu().squeeze(0).numpy().transpose(1,2,0),cv2.COLOR_RGB2BGR))
        output=srcnn(output)
        output=output.cpu().squeeze(0).numpy().transpose(1,2,0)
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




if __name__ == "__main__":
    set_style("8.jpg")
    video_mod()