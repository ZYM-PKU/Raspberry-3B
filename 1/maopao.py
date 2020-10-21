import random # 导入random 模块

data = [] # 定义一个空的列表data

for i in range(0,10): # 在列表data 插入10 个0~99 之间的随机数
    data.append(random.randint(0,99))

print("Unordered list:")
print(data) # 打印列表


#冒泡排序：
count = len(data) # 获取列表的长度
for i in range(count-1,0,-1):
    for j in range(0,i):
        if data[j]<data[j+1]:
            data[j],data[j+1]=data[j+1],data[j]#交换

print("Ordered list:")
print(data)


