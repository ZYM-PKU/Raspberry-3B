import RPi.GPIO as GPIO
import time


KEY = 20
GPIO.setmode(GPIO.BCM)
GPIO.setup(KEY, GPIO.IN,GPIO.PUD_UP) # 上拉电阻
print("Key Test Program")


def my_callback(ch):
    print("KEY PRESS")


GPIO.add_event_detect(KEY, GPIO.FALLING, callback=my_callback, bouncetime=200)
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    GPIO.cleanup()
    