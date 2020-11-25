from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2 # 导入需要的库

camera = PiCamera()
camera.resolution = (640, 480) # 设置分辨率
camera.framerate = 32 # 设置帧率
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.1) # 等待摄像头模块初始化


locked=False
prev_info=[0,0,0]

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = frame.array
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # 色彩变换
    blur = cv2.blur(grey,(5,5)) # 过滤噪声
    circles = cv2.HoughCircles(blur, # 识别圆形
    method=cv2.HOUGH_GRADIENT,dp=1,minDist=200,
    param1=100,param2=33,minRadius=30,maxRadius=175)
    locked=False
    if circles is not None: # 识别到圆形
        for i in circles [0,:]: # 画出识别的结果
            if  not locked or sum([p**2 for p in (i-prev_info)[0:2]])**0.5<20:
                cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
                cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
                prev_info=[i[0],i[1],i[2]]
                locked=True
    else: 
        locked=False

    cv2.imshow('Detected',frame) # 显示识别图像
    key = cv2.waitKey(1) & 0xFF # 等待按键
    rawCapture.truncate(0) # 准备下一副图像
    if key == ord("q"):
        break