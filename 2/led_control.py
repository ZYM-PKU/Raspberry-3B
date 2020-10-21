import RPi.GPIO as GPIO
import time

LED = 26
KEY = 20
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED,GPIO.OUT)
GPIO.setup(KEY, GPIO.IN,GPIO.PUD_UP) # 上拉电阻


freq=1#闪烁频率
state=0#当前状态
pulse_time=time.time()#记录时间轴上最近一次上跳沿的时刻
p = GPIO.PWM(LED,freq)




def callback1():
    global state
    state=1 if state==0 or 0#转换状态
    GPIO.output(LED,state)

def control1():
    GPIO.output(LED,state)
    add_event_detect(KEY, GPIO.RISING, callback=callback1, bouncetime=200)
    try:
        while True:
            pass
    except KeyboardInterrupt:
        p.stop()
        GPIO.cleanup()

def callback2():
    global pulse_time,freq
    t=time.time()
    if t-pulse_time<0.5:
        freq=1#重置
        p.setChangeFrequency(50)#高频输出（相当于不闪烁）
    else:
        freq=2*freq#频率加倍
        p.setChangeFrequency(freq)

    pulse_time=t#记录上跳沿时刻




def control2():
    global p
    p.start(100)
    add_event_detect(KEY, GPIO.RISING, callback=callback2, bouncetime=200)



if __name__ == "__main__":
    control1()
    try:
        while True:
            pass
    except KeyboardInterrupt:
        p.stop()
        GPIO.cleanup()

