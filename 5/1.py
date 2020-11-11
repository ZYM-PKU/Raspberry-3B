import smbus
import time # 包含相关库文件
from datetime import datetime
address = 0x68
register = 0x00
bus = smbus.SMBus(1) # 初始化i2c Bus

# FixTime 定义为2019 年6 月12 日18 时
FixTime = [0x00,0x00,0x13,0x03,0x11,0x11,0x20]

def gettime():
    dt = datetime.now()
    return dt.hour,dt.minute,dt.second


def ds3231SetTime():
    h,m,s=gettime()
    h,m,s=16*(h//10)+h%10,16*(m//10)+m%10,16*(s//10)+s%10
    FixTime = [s,m,h,0x03,0x11,0x11,0x20]
    bus.write_i2c_block_data(address,register,FixTime)
def ds3231ReadTime():
    datas= bus.read_i2c_block_data(address,register,7)
    datas=[(data//16)*10+data%16  for data in datas]
    return datas

ds3231SetTime() # 设置时间

while True:
    t=ds3231ReadTime()
    print(t) # 读出时间
    print(f"Time: 20{t[6]}year {t[5]}month {t[4]}day  {t[2]}h {t[1]}m {t[0]}s")
    print("-------------------------------------------------")
    time.sleep(1)

