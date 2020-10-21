import numpy as np
import RPi.GPIO as GPIO
import time

KEY = 20
LED = 26
GPIO.setmode(GPIO.BCM)
GPIO.setup(KEY, GPIO.IN,GPIO.PUD_UP) # 上拉电阻
GPIO.setup(LED,GPIO.OUT)
GPIO.output(LED,0)


m1,m2,m3=1,2,3#目标参数设置


length = 2000#数据集大小
x1 = np.random.rand(1,length) * 10 # 产生均匀分布的x1
x2 = np.random.rand(1,length) * 10

x=np.concatenate((x1,x2),axis=0)#垂直堆叠，向量化输入
y = x1 * m1 + x2 * m2 + m3 + np.random.randn(1,length) # y 要加上随机噪声


learning_rate=0.02#学习率
iter_times=100#训练次数
press_times=0#按键次数

#参数矩阵初始化
w=np.random.rand(1,2)
b=np.random.rand(1,1)



def callback(ch):
    global w,b,x,y,press_times,learning_rate
    if learning_rate>0.001:learning_rate-=0.001

    press_times+=1
    loss_list=[]

    for epoch in range(iter_times):

        res=np.dot(w,x)+b#矩阵乘法+broadcast

        loss=(1/length)*np.sum((res-y)**2)#计算损失
        loss_list.append(loss)
        #print(f"loss: {loss}")#打印损失

        #求导
        dloss=(1/length)*(res-y)
        db=np.sum(dloss)
        dw=np.dot(dloss,x.T)

        #梯度下降
        w-=learning_rate*dw
        b-=learning_rate*db
    
    average_loss=sum(loss_list)/len(loss_list)
    if abs(b[0][0]-m3)<0.1:
        GPIO.output(LED,1)#点亮灯泡

    print(f"press_times:  {press_times}")
    print(f"average_loss: {average_loss}")#打印平均损失
    print(f"w:  {w}")
    print(f"b:  {b}")




if __name__ == "__main__":
    print("Press to start...")
    GPIO.add_event_detect(KEY, GPIO.RISING, callback=callback, bouncetime=200)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        GPIO.cleanup()

