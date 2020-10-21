import RPi.GPIO as GPIO
import time,threading

LED = 26
KEY = 20
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED,GPIO.OUT)
GPIO.setup(KEY, GPIO.IN,GPIO.PUD_UP) # 上拉电阻


freq=1#闪烁频率
state=0#当前状态
pulse_time=time.time()#记录时间轴上最近一次上跳沿的时刻
p = GPIO.PWM(LED,freq)




def callback1(ch):
    global state
    state=1 if state==0 else 0#转换状态
    GPIO.output(LED,state)

def control1():
    GPIO.output(LED,state)
    GPIO.add_event_detect(KEY, GPIO.RISING, callback=callback1, bouncetime=100)


def callback2(ch):
    global pulse_time,freq
    t=time.time()
    if t-pulse_time<0.5:
        freq=1#重置
        p.ChangeDutyCycle(100)#dc=100
    else:
        timer=threading.Timer(0.5,change_freq)
        timer.start()
    pulse_time=t#记录上跳沿时刻

def change_freq():
    global pulse_time,freq
    t=time.time()
    if t-pulse_time>=0.5:
        p.ChangeDutyCycle(50)
        freq=2*freq#频率加倍
        p.ChangeFrequency(freq)

    

def control2():
    global p
    p.start(50)
    GPIO.add_event_detect(KEY, GPIO.RISING, callback=callback2, bouncetime=200)



if __name__ == "__main__":
    control2()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        p.stop()
        GPIO.cleanup()

