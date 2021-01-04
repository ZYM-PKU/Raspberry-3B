from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import os
import cv2 # 导入需要的库
#import torch
import socket
import struct
import smbus
import SSD1306
import spidev as SPI
import numpy as np
import RPi.GPIO as GPIO
from PIL import Image, ImageFile
from datetime import datetime
from threading import Thread
from PIL import Image,ImageDraw,ImageFont

bus = smbus.SMBus(1)
disp=SSD1306.SSD1306(rst=19,dc=16,spi=SPI.SpiDev(0,0))
disp.begin()
disp.clear()

print("Initializing...")

#初始化摄像头
presolution=(480,270)#照片清晰度
vresolution=(200,160)#视频帧清晰度
camera = PiCamera()
camera.resolution = vresolution # 设置分辨率
camera.framerate = 90 # 设置帧率
rawCapture = PiRGBArray(camera, size=vresolution)
time.sleep(0.1) # 等待摄像头模块初始化


PATH = os.path.dirname(__file__)
address = 0x20
video_mode=False#视频模式


photo_count=1#照片数
curr_style=0
fps=0#帧速率


#connect
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Connecting...")
    s.connect(('192.168.137.1', 8088))
    print("Connected")
except socket.error as msg:
    print(msg)

KEY = 20
GPIO.setmode(GPIO.BCM)
GPIO.setup(KEY, GPIO.IN,GPIO.PUD_UP) # 上拉电阻




def photo_capture():
    '''照相并风格化'''
    global photo_count,camera,s
    

    #cv2.destroyAllWindows()

    save_path=os.path.join(PATH,f'save/{curr_style}capture_{photo_count}.jpg')
    camera.resolution = presolution
    camera.capture(save_path, use_video_port = False)#获取一张照片
    imarray=cv2.imread(save_path)
    cv2.imshow("captured",imarray)
    cv2.moveWindow("captured",0,100)
    simarray=cv2.imread(os.path.join(PATH,f"{curr_style+1}.jpg"))

    cv2.namedWindow("style",0)
    cv2.resizeWindow("style",presolution[0],presolution[1])
    cv2.imshow("style",simarray)
    cv2.moveWindow("style",0,presolution[1]+100)

    
    try:

        fhead = struct.pack(b'128sl', bytes(os.path.basename(save_path), encoding='utf-8'), os.stat(save_path).st_size)
        s.send(fhead)
        #print('client filepath: {0}'.format(filepath))

        fp = open(save_path, 'rb')
        while 1:
            data = fp.read(1024)
            if not data:
                #print('{0} 发送成功...'.format(save_path))
                break
            s.send(data)

        time.sleep(0.01)

        fileinfo_size = struct.calcsize('128sl')
        buf = s.recv(fileinfo_size)
        if buf:
            filename, filesize = struct.unpack('128sl', buf)
            fn = filename.decode().strip('\x00')
            #PC端图片保存路径
            new_filename = os.path.join(PATH, 'save/'+fn)

            recvd_size = 0
            fp = open(new_filename, 'wb')

            while not recvd_size == filesize:
                if filesize - recvd_size > 1024:
                    data = s.recv(1024)
                    recvd_size += len(data)
                else:
                    data = s.recv(1024)
                    recvd_size = filesize
                fp.write(data)
            fp.close()
    except:
        print("packet loss")

    imarray=cv2.imread(os.path.join(PATH,'save/'+fn))

    cv2.imshow("stylized",imarray)
    cv2.moveWindow("stylized",presolution[0],100)
    cv2.waitKey(1000)
    
    print("Captured")
    photo_count+=1

    s.close()



 


def video_mod():
    global s,fps
    
    camera.resolution = vresolution # 设置分辨率
    
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        

        tic=time.time()
        imarray = frame.array
        save_path=os.path.join(PATH,f'{curr_style}capture.jpg')
        cv2.imwrite(save_path,imarray)
        cv2.imshow("origin",imarray)
        #cv2.moveWindow("origin",500,200)

        simarray=cv2.imread(os.path.join(PATH,f"{curr_style+1}.jpg"))
        cv2.namedWindow("style",0)
        #cv2.resizeWindow("style",vresolution[0],vresolution[1])
        cv2.moveWindow("style",500,200+vresolution[1])
        #cv2.imshow("style",simarray)
        
        fn=''
        
        try:

            fhead = struct.pack(b'128sl', bytes(os.path.basename(save_path), encoding='utf-8'), os.stat(save_path).st_size)
            s.send(fhead)
            #print('client filepath: {0}'.format(savepath))

            fp = open(save_path, 'rb')
            while 1:
                data = fp.read(1024)
                if not data:
                    #print('{0} 发送成功...'.format(filepath))
                    break
                s.send(data)

            #time.sleep(0.08)

            fileinfo_size = struct.calcsize('128sl')
            buf = s.recv(fileinfo_size)
            if buf:
                filename, filesize = struct.unpack('128sl', buf)
                fn = filename.decode().strip('\x00')
                #PC端图片保存路径
                new_filename = os.path.join(PATH, fn)
    
                recvd_size = 0
                fp = open(new_filename, 'wb')
    
                while not recvd_size == filesize:
                    if filesize - recvd_size > 1024:
                        data = s.recv(1024)
                        recvd_size += len(data)
                    else:
                        data = s.recv(1024)
                        recvd_size = filesize
                    fp.write(data)
                fp.close()
            imarray=cv2.imread(os.path.join(PATH,fn))
            cv2.imshow("stylized",imarray)
            cv2.moveWindow("stylized",500+presolution[0],200)
        except:
            print("packet loss")

        
        
        s.close()
        key = cv2.waitKey(1) & 0xFF # 等待按键
        rawCapture.truncate(0) # 准备下一副图像
        toc=time.time()
        fps=1.0/(toc-tic)

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('192.168.137.1', 8088))


       

def gettime():
    dt = datetime.now()
    hour = '0'+str(dt.hour) if len(str(dt.hour)) == 1 else str(dt.hour)
    minute = '0'+str(dt.minute) if len(str(dt.minute)) == 1 else str(dt.minute)
    second = '0'+str(dt.second) if len(str(dt.second)) == 1 else str(dt.second)
    timestr = hour+':'+minute+':'+second

    return timestr

def dispoled():
    '''OLED显示'''
    global curr_style,fps
    while True:

        nowtime = gettime()

        logo = Image.open(os.path.join(PATH, 'p128.png')).resize(
            (32, 32), Image.ANTIALIAS).convert('1')  # logo
        img = Image.new('1', (disp.width, disp.height), 'black')  # final_img
        img.paste(logo, (0, 0, logo.size[0], logo.size[1]))

        font = ImageFont.truetype("comicsansms.ttf", 13)
        draw = ImageDraw.Draw(img)
        draw.bitmap((0, 0), img, fill=1)
        draw.text((64, 0), nowtime, font=font, fill=255)
        draw.text((40, 20), f'Style:{curr_style}', font=font, fill=255)
        draw.text((0, 40), f'frame rate: {round(fps,2)}fps', font=font, fill=255)

        disp.clear()
        disp.image(img)
        disp.display()
        time.sleep(0.1)
    

def change_style():
    global curr_style
    while True:
        value = bus.read_byte(address) | 0xF0
        if ((value | 0xFD) != 0xFF):
            curr_style=(curr_style+1)%9
        time.sleep(0.1)



if __name__ == "__main__":
    print("LISTENING...")
    t=Thread(target=dispoled)
    t.start()
    t1=Thread(target=change_style)
    t1.start()

    while True:
        
        if not GPIO.input(KEY):
            photo_capture()
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(('192.168.137.1', 8088))
            if not GPIO.input(KEY):
                cv2.destroyAllWindows()
                video_mod()
        time.sleep(0.1)