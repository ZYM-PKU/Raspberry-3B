import time
import spidev as SPI
import SSD1306
from PIL import Image,ImageDraw,ImageFont # 调用相关库文件
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
import random
import numpy as np
import RPi.GPIO as GPIO
KEY = 20
GPIO.setmode(GPIO.BCM)
GPIO.setup(KEY, GPIO.IN,GPIO.PUD_UP)

RST = 19
DC = 16
bus = 0
device = 0 # 树莓派管脚配置
disp = SSD1306.SSD1306(rst=RST,dc=DC,spi=SPI.SpiDev(bus,device))

disp.begin()
disp.clear()
disp.display()

images_and_predictions=[]


def train():
    global images_and_predictions

    digits = datasets.load_digits()

    # To apply a classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001)
    # We learn the digits on the first half of the digits
    classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])
    # Now predict the value of the digit on the second half:
    expected = digits.target[n_samples // 2:]
    predicted = classifier.predict(data[n_samples // 2:])


    images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
    print("train completed.")



for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)
plt.show()

def showimg(ch):
    

    kk,pre=random.choice(images_and_predictions)
    digit = Image.fromarray((kk*8).astype(np.uint8), mode='L').resize((48,48)).convert('1')

    img = Image.new('1',(disp.width,disp.height),'black')
    img.paste(digit, (40, 16, digit.size[0]+40, digit.size[1]+16))
    disp.clear()
    disp.image(img)
    disp.display()

    print(f"predictions:  {pre}")


def main():
    images_and_predictions=train()
    GPIO.add_event_detect(KEY, GPIO.RISING, callback=showimg, bouncetime=200)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()


