import pyaudio,wave
import RPi.GPIO as GPIO
import time,os
from scipy.io import wavfile
import joblib
from hmmlearn import hmm
from python_speech_features import mfcc
import numpy as np

PATH = os.path.dirname(__file__)
KEY = 20
GPIO.setmode(GPIO.BCM)
GPIO.setup(KEY, GPIO.IN,GPIO.PUD_UP) # 上拉电阻

CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100


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

wavdict={11:"open1.wav",12:"open2.wav",13:"open3.wav",14:"open4.wav",15:"open5.wav",16:"open6.wav",17:"open7.wav",
21:"close1.wav",22:"close2.wav",23:"close3.wav",24:"close4.wav",25:"close5.wav",26:"close6.wav",27:"close7.wav",
31:"up1.wav",32:"up2.wav",33:"up3.wav",34:"up4.wav",35:"up5.wav",36:"up6.wav",37:"up7.wav",
41:"down1.wav",42:"down2.wav",43:"down3.wav",44:"down4.wav",45:"down5.wav",46:"down6.wav",47:"down7.wav",
51:"noise1.wav",52:"noise2.wav",53:"noise3.wav",54:"noise4.wav",55:"noise5.wav",56:"noise6.wav",57:"noise7.wav",
}


def disp1(string):
    '''显示helloworld'''

    font = ImageFont.truetype("comicsansms.ttf", 20)
    image = Image.new('RGB', (disp.width, disp.height), 'black').convert('1')
    draw = ImageDraw.Draw(image)
    draw.bitmap((0, 0), image, fill=1)
    draw.text((10, 20), string, font=font, fill=255)
    disp.image(image)
    disp.display()  # 显示图片



def compute_mfcc(file):
    fs, audio = wavfile.read(os.path.join(PATH,file))
    mfcc_feat = mfcc(audio)
    return mfcc_feat

class Model():
    def __init__(self, CATEGORY=None, n_comp=3, n_mix = 3, cov_type='diag', n_iter=1000):
        super(Model, self).__init__()
        self.CATEGORY = CATEGORY
        self.category = len(CATEGORY)
        self.n_comp = n_comp
        self.n_mix = n_mix
        self.cov_type = cov_type
        self.n_iter = n_iter
        # 关键步骤，初始化models，返回特定参数的模型的列表
        self.models = []
        for k in range(self.category):
            model = hmm.GMMHMM(n_components=self.n_comp, n_mix = self.n_mix,
            covariance_type=self.cov_type, n_iter=self.n_iter)
            self.models.append(model)


    def save(self,path=os.path.join(PATH,"models.pkl")):
        joblib.dump(self.models,path)

    def load(self,path=os.path.join(PATH,"models.pkl")):
        self.models=joblib.load(path)

    def train(self):
        for k in range(self.category):
            model=self.models[k]
            for x in wavdict:
                if x//10==k+1:
                    mfcc_feat=compute_mfcc(os.path.join(PATH,wavdict[x]))
                    model.fit(mfcc_feat)

    
    def test(self,path):
        result=[]
        for k in range(self.category):
            model=self.models[k]
            mfcc_feat=compute_mfcc(os.path.join(PATH,path))
            re=model.score(mfcc_feat)
            result.append(re)
        result=self.CATEGORY[np.array(result).argmax()]

        return result

def record_audio(path,record_second):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK)
    wf = wave.open(os.path.join(PATH,path), 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    print("*recording...")
    for i in range(int(RATE*record_second/CHUNK)):
        data=stream.read(CHUNK)
        wf.writeframes(data)
    print("recond end.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()

def record():
    print("KEY PRESS")
    record_audio(f"test.wav",1.5)

def train():
    print("start training...")
    model=Model(["open","close","up","down","noise"])
    model.train()
    model.save()
    print("well trained.")

def test():
    print("Start testing...")
    model=Model(["open","close","up","down","noise"])
    model.load()
    try:
        while True:
            print("KEY PRESS DETECTED")
            record()
            result=model.test('test.wav')
            print(f"You said:{result}")
            disp.clear()
            disp1(result)
            time.sleep(0.1)


    except KeyboardInterrupt:
        GPIO.cleanup()






if __name__ == "__main__":
    test()




