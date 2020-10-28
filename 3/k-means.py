import time,os,glob,random
import numpy as np
import matplotlib.pyplot as plt


PATH="/sys/bus/w1/devices/28-0000096582fc/w1_slave"#温度文件路径




time1,time2=5,7
aver_temp1,aver_temp2=22,28
sigma1,sigma2=0.2,0.4

temp1=np.random.randn(time1*2*30)*sigma1+aver_temp1
temp2=np.random.randn(time2*2*30)*sigma2+aver_temp2

temp=np.append(temp1,temp2)#组合数据
np.random.shuffle(temp)#打乱数据

plt.hist(temp,50,histtype='bar',facecolor='yellowgreen',alpha=0.75)
plt.show()

#k-means
core1,core2=random.randint(20,24),random.randint(26,30)
list1,list2=[],[]
prev1,prev2=core1,core2
while True:
      for data in temp:
          dis1=abs(data-core1)
          dis2=abs(data-core2)
          if dis1<dis2:
              list1.append(data)
          else:
              list2.append(data)
      try:
          core1=sum(list1)/len(list1)
      except:
          core1=random.randint(20,30)
      try:
          core2=sum(list2)/len(list2)
      except:
          core2=random.randint(20,30)
      
      if core1==prev1 and core2==prev2:
          print(f"temp1: {core1}   temp2: {core2}")
          print(f"time1: {len(list1)/60}h  time2: {len(list2)/60}h")
          break
      else:
          prev1,prev2=core1,core2
          list1.clear()
          list2.clear()
          
#get_temp
t=0
while True:
   with open(PATH,'r') as f:
        lines=f.readlines()
        msg=lines[0][-4:-1]
        if msg=='YES':
            t=round(int((lines[-1].split('='))[-1])/1000,3)#读取温度，保留三位小数
            break
        
print(f"temp: {t} *C")
dis1,dis2=abs(t-core1),abs(t-core2)
if dis1<dis2:
    print("A is in room.")
else:
    print("B is in room.")

          


