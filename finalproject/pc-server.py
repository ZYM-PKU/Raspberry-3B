import time
import os
import cv2 # 导入需要的库
import torch
import socket
import struct
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
pre_style=0#当前风格


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

encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]



#初始化风格特征
def test_transform(size):
    transform_list = []
    transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform



def set_style(number):
    global fbnet
    #设置风格
    stylepath = os.path.join(PATH,f"{number+1}.jpg")
    styleimg = Image.open(str(stylepath)).convert('RGB')
    styleimg = tt(styleimg).unsqueeze(0).cuda()

    sfeature=[]
    layers=[31,4,11,18,31]
    for i in range(31):
        styleimg = encoder[i](styleimg)
        if i==3 or i==10 or i==17 or i==30:sfeature.append(styleimg)

    fbnet.set_style_feature(sfeature)



tt=transforms.ToTensor()
transform = test_transform(SCALE[1])
up2=nn.Upsample(scale_factor=2, mode='bilinear')
srcnn=SRCNN(test=True).eval().cuda()
srcnn.load_state_dict(torch.load(os.path.join(PATH,"model/srnet_x2.pth")))
set_style(pre_style)


 
 
def socket_service():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #IP地址留空默认是本机IP地址
        s.bind(('', 8088))
        s.listen(7)
    except socket.error as msg:
        print(msg)
        sys.exit(1)
 
    print("连接开启，等待传图...")
	
    while True:
        sock, addr = s.accept()
        try:
            deal_data(sock, addr)
        except:
            print("interrupted")
 
    s.close()
 
 
def deal_data(sock, addr):
    #print("成功连接上 {0}".format(addr))
    global pre_style

    fileinfo_size = struct.calcsize('128sl')
    buf = sock.recv(fileinfo_size)
    fn=''
    if buf:
        filename, filesize = struct.unpack('128sl', buf)
        fn = filename.decode().strip('\x00')
        #PC端图片保存路径
        new_filename = os.path.join(PATH, fn)

        recvd_size = 0
        fp = open(new_filename, 'wb')

        while not recvd_size == filesize:
            if filesize - recvd_size > 1024:
                data = sock.recv(1024)
                recvd_size += len(data)
            else:
                data = sock.recv(1024)
                recvd_size = filesize
            fp.write(data)
        fp.close()

    tic=time.time()

    curr_style=int(fn[0])
    if curr_style!=pre_style:
        set_style(curr_style)
        pre_style=curr_style

    fn=fn[:-4]
    pos=fn.find("_")
    savepath = os.path.join(PATH,f"{curr_style}stylized.jpg")


    if pos!=-1:
        number=int(fn[pos+1:])
        savepath = os.path.join(PATH,f"{curr_style}stylized_{number}.jpg")

    try:

        image = Image.open(new_filename).convert('RGB')
        image = tt(image).unsqueeze(0).cuda()
        image = up2(image)
        image = srcnn(image)
        output=fbnet(image,alpha=1.0,lamda=1.0,require_loss=False)
        output=output.cpu()
        save_image(output,savepath)

    except:
        print("package loss")

    
    toc=time.time()
    print(f"Time cost:{toc-tic}s")
    try:
        fhead = struct.pack(b'128sl', bytes(os.path.basename(savepath), encoding='utf-8'), os.stat(savepath).st_size)
        sock.send(fhead)
        #print('client filepath: {0}'.format(filepath))

        fp = open(savepath, 'rb')
        while 1:
            data = fp.read(1024)
            if not data:
                #print('{0} 发送成功...'.format(savepath))
                break
            sock.send(data)

        sock.close()
    except:
        print("error")
        
    
 
 
if __name__ == '__main__':
    socket_service()

