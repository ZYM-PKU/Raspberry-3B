import RPi.GPIO as GPIO

KEY = 20
GPIO.setmode(GPIO.BCM)
GPIO.setup(KEY, GPIO.IN,GPIO.PUD_UP) # 上拉电阻
print("Key Test Program")


def my_callback(ch):
    print("KEY PRESS")


add_event_detect(KEY, GPIO.RISING, callback=my_callback, bouncetime=200)