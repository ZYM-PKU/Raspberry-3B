import RPi.GPIO as GPIO
import time,os,glob

LED = 26
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED,GPIO.OUT)
GPIO.output(LED,0)

PATH="/sys/bus/w1/devices/28-0000096582fc/w1_slave"#温度文件路径




while True:
    try:
        #os.system('clear')#清屏
        t=0
        with open(PATH,'r') as f:
            lines=f.readlines()
            msg=lines[0][-4:-1]
            if msg=='YES':
                t=str(round(int((lines[-1].split('='))[-1])/1000,3))#读取温度，保留三位小数
            else:
                t="N/A"

        #输出
        print("-------TEMPERATURE-------")
        print()
        print('     '+str(t)+'  *C')
        print()
        print("-------------------------")

        #报警
        try:
           if  float(t)>28:
               GPIO.output(LED,1)#温度报警
           else:
               GPIO.output(LED,0)
        except:pass

        time.sleep(1)
    
    except KeyboardInterrupt:
        GPIO.cleanup()
        break
