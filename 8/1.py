import pyaudio,wave
import RPi.GPIO as GPIO
import time,os

#sudo arecord test.wav -D hw:2,0 -f S16_LE -r 44100 -d 5
#aplay -D hw:1,0 test.wav
KEY = 20
GPIO.setmode(GPIO.BCM)
GPIO.setup(KEY, GPIO.IN,GPIO.PUD_UP) # 上拉电阻

CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

PATH = os.path.dirname(__file__)
counting=1


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
        if GPIO.input(KEY):break
    print("recond end.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()

def record():
    global counting
    print("KEY PRESS")
    record_audio(f"noise{counting}.wav",30)
    counting+=1
    





#GPIO.add_event_detect(KEY, GPIO.FALLING, callback=record, bouncetime=200)
try:
    while True:
        if not GPIO.input(KEY):
            print("KEY PRESS DETECTED")
            record()
        time.sleep(0.1)
except KeyboardInterrupt:
    GPIO.cleanup()