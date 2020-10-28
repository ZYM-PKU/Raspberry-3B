import time,os,glob
import numpy as np
import matplotlib.pyplot as plt

PATH="/sys/bus/w1/devices"#温度文件路径

filename=glob.glob(PATH+"/*")[0]#读取温度文件


time1,time2=5,7
aver_temp1,aver_temp2=26,28
sigma1,sigma2=1,2

temp1=np.random.randn(time1*2)*sigma1+aver_temp1
temp2=np.random.randn(time2*2)*sigma2+aver_temp2

temp=np.append(temp1,temp2)#组合数据
np.random.shuffle(temp)#打乱数据

plt.hist(temp,10,histtype='bar',facecolor='yellowgreen',alpha=0.75)
plt.show()