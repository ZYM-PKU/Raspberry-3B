import RPi.GPIO as GPIO
import time

LED = 26
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED,GPIO.OUT)



def display1():
    state=0
    try:
        while True:
            state=1 if state==0 else 0#变换状态
            GPIO.output(LED,state)
            time.sleep(0.2)
    except KeyboardInterrupt:
        GPIO.cleanup()

def display2():
    p = GPIO.PWM(LED,2.5)
    p.start(50)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        p.stop()
        GPIO.cleanup()



def display3():

    p = GPIO.PWM(LED,50)
    p.start(0)

    try:
        while True:
            for dc in range(0,101,5):
                p.ChangeDutyCycle(dc)
                time.sleep(0.05)
            for dc in range(100,-1,-5):
                p.ChangeDutyCycle(dc)
                time.sleep(0.05)

    except KeyboardInterrupt:
        p.stop()
        GPIO.cleanup()


if __name__ == "__main__":
    display3()