from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2 # 导入需要的库

camera = PiCamera()
camera.resolution = (640, 480) # 设置分辨率
camera.framerate = 32 # 设置帧率
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.1) # 等待摄像头模块初始化

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    cv2.imshow("Frame", image) # 显示图像
    key = cv2.waitKey(1) & 0xFF # 等待按键
    rawCapture.truncate(0) # 准备下一副图像
    if key == ord("q"):
        break