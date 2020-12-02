import pyaudio,wave
import RPi.GPIO as GPIO
import time,os


KEY = 20
GPIO.setmode(GPIO.BCM)
GPIO.setup(KEY, GPIO.IN,GPIO.PUD_UP) # 上拉电阻

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100

PATH = os.path.dirname(__file__)
counting=1

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK)

def record(ch):
    print("KEY PRESS")
    pyaudio.Stream.write()
    
def read(ch):
    print("RELEASE")
    pyaudio.Stream.close()
    wf = wave.open(os.path.join(PATH,f"record{counting}.wav"), 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(data)
    counting+=1




GPIO.add_event_detect(KEY, GPIO.RISING, callback=record, bouncetime=200)
GPIO.add_event_detect(KEY, GPIO.FALLING, callback=read, bouncetime=200)
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    GPIO.cleanup()