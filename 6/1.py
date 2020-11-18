import smbus
import time
import os
import spidev as SPI
import SSD1306
from PIL import Image, ImageDraw, ImageFont  # 调用相关库文件
from datetime import datetime
from threading import Thread

PATH = os.path.dirname(__file__)
RST = 19
DC = 16
bus = 0
device = 0  # 树莓派管脚配置
disp = SSD1306.SSD1306(rst=RST, dc=DC, spi=SPI.SpiDev(bus, device))

disp.begin()
disp.clear()



address = 0x48
A0 = 0x40

bus = smbus.SMBus(1) # 初始化 i2c Bus
bus.write_byte(address,A0)


value=0




def gettime():
    dt = datetime.now()
    hour = '0'+str(dt.hour) if len(str(dt.hour)) == 1 else str(dt.hour)
    minute = '0'+str(dt.minute) if len(str(dt.minute)) == 1 else str(dt.minute)
    second = '0'+str(dt.second) if len(str(dt.second)) == 1 else str(dt.second)
    timestr = hour+':'+minute+':'+second

    return timestr


def disp1():
    '''显示helloworld'''
    global value
    font = ImageFont.truetype("comicsansms.ttf", 15)
    while True:

        value = bus.read_byte(address) # 循环读出
        image = Image.new('RGB', (disp.width, disp.height), 'black').convert('1')
        draw = ImageDraw.Draw(image)
        draw.bitmap((0, 0), image, fill=1)
        draw.text((10, 20), f"Current Voltage: \n{round(value/256.0*3.3,2)}V  ", font=font, fill=255)
        disp.clear()
        disp.image(image)
        disp.display()
        time.sleep(0.1)



def led():
    global value
    light=0
    delta=20
    while True:

        light+=delta
        if light>255:
            delta=-delta
            light=255
            
        if light<0:
            delta=-delta
            light=0
        bus.write_byte_data(address, A0, light)
        time.sleep(0.26-0.1*(value/100))


if __name__ == "__main__":
    thread1=Thread(target=disp1)
    thread2=Thread(target=led)
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()