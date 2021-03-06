import time
import os
import spidev as SPI
import SSD1306
from PIL import Image, ImageDraw, ImageFont  # 调用相关库文件
from datetime import datetime


PATH = os.path.dirname(__file__)
RST = 19
DC = 16
bus = 0
device = 0  # 树莓派管脚配置
disp = SSD1306.SSD1306(rst=RST, dc=DC, spi=SPI.SpiDev(bus, device))

disp.begin()
disp.clear()


def gettime():
    dt = datetime.now()
    hour = '0'+str(dt.hour) if len(str(dt.hour)) == 1 else str(dt.hour)
    minute = '0'+str(dt.minute) if len(str(dt.minute)) == 1 else str(dt.minute)
    second = '0'+str(dt.second) if len(str(dt.second)) == 1 else str(dt.second)
    timestr = hour+':'+minute+':'+second

    return timestr


def disp1():
    '''显示helloworld'''

    font = ImageFont.truetype("comicsansms.ttf", 20)
    image = Image.new('RGB', (disp.width, disp.height), 'black').convert('1')
    draw = ImageDraw.Draw(image)
    draw.bitmap((0, 0), image, fill=1)
    draw.text((10, 20), 'Hello World!', font=font, fill=255)
    disp.image(image)
    disp.display()  # 显示图片


def disp2():
    '''显示时钟'''
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
        draw.text((32, 15), "Count down of ", font=font, fill=255)
        draw.text((50, 30), "mid-term:", font=font, fill=255)

        tardate = datetime(2020, 11, 9)
        nowdate = datetime.now()
        delta = tardate-nowdate
        days = delta.days
        seconds = delta.seconds
        hours = seconds//3600
        seconds = seconds % 3600
        minutes = seconds//60
        seconds = seconds % 60

        draw.text((0, 45), f"{days}d  {hours}hour {minutes}min {seconds}s", 
        font=font, fill=255)

        disp.clear()
        disp.image(img)
        disp.display()
        time.sleep(0.1)


if __name__ == "__main__":
    disp2()
