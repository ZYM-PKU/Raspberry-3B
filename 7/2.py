from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2 # 导入需要的库
import numpy as np

camera = PiCamera()
camera.resolution = (640, 480) # 设置分辨率
camera.framerate = 32 # 设置帧率
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.1) # 等待摄像头模块初始化

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = frame.array
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) # 转换颜色空间
    print(hsv[240][320])
    # 通过颜色设计模板
    image_mask=cv2.inRange(hsv,np.array([0,0,0]), np.array([50,255,255]))
    # 计算输出图像
    output=cv2.bitwise_and(frame,frame,mask=image_mask)
    cv2.imshow('Original',frame) # 显示原始图像
    cv2.imshow('Output',output) # 显示输出图像
    key = cv2.waitKey(1) & 0xFF # 等待按键
    rawCapture.truncate(0) # 准备下一副图像
    if key == ord("q"):
        break