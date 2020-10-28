import RPi.GPIO as GPIO
import time,os,glob

LED = 26
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED,GPIO.OUT)
GPIO.output(LED,0)

PATH="/sys/bus/w1/devices"#温度文件路径

filename=glob.glob(PATH+"/*")[0]#读取温度文件


while True:
    os.system('cls')#清屏
    t=0
    with open(filename,'r') as f:
        lines=f.readlines()
        t=round(int((f[-1].split('='))[-1])/1000,3)#读取温度，保留三位小数
    
    #输出
    print("-------TEMPERATURE-------")
    print()
    print('     '+t+'  *C')
    print()
    print("-------------------------")

    #报警
    if t>30:
        GPIO.output(LED,1)#温度报警
    else:
        GPIO.output(LED,0)

    time.sleep(1)

