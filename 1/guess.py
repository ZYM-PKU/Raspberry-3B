import numpy as np
import random



#定义双层简单神经网络
X=np.ones((3,1))
W_1,W_2= np.random.randn(3,3),np.random.randn(3,3)
B_1,B_2= np.random.randn(3,1),np.random.randn(3,1)
Y_1,Y_2,Y= [np.zeros((3,1)) for i in range(3)]


lr=0.1#学习率




def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return max(0,x)


def intell_guess():
    '''带有一定智能的猜拳函数'''
    global Y_1,Y_2,W_1,W_2,B_1,B_2

    Y_1=np.dot(W_1,X)+B_1
    Y_1=sigmoid(Y_1)
    Y_2=np.dot(W_2,Y_1)+B_2
    Y_2=sigmoid(Y_2)#正则化

    return np.argmax(Y_2)#返回最大相应
    

def random_guess():
    '''随机猜拳函数'''
    return random.choice([0,1,2,2,2])




def main1():
    global Y_1,Y_2,W_1,W_2,B_1,B_2
    user = 0
    while True:
        try:
            user = int(input("请输入:剪刀(0) 石头(1) 布(2)退出(3):"))
        except:
            print("Illegal Iuput! Number Only!")
            continue

        #根据用户输入得到目标相应（3*1矩阵）
        if user == 3: break
        elif user==0:
            Y=np.array([[0,1,0]]).T
        elif user==1:
            Y=np.array([[0,0,1]]).T
        elif user==2:
            Y=np.array([[1,0,0]]).T

        computer = intell_guess()#猜拳

        print(Y_2)

        #求导
        dZ_2=Y_2-Y
        dW_2=np.dot((dZ_2),Y_1.T)
        dB_2=dZ_2
        dZ_1=np.dot(W_2.T,dZ_2)*(Y_1*(1-Y_1))
        dW_1=np.dot((dZ_1),X.T)
        dB_1=dZ_1

        #梯度下降
        W_2-=lr*dW_2
        W_1-=lr*dW_1
        B_2-=lr*dB_2
        B_1-=lr*dB_1


        if (user == 0 and computer == 2) or (user == 1 and computer == 0) or (user == 2 and computer == 1):
            print(f"电脑出了{computer}，你赢了")
        elif computer == user:
            print(f"电脑出了{computer}，平局")
        else:
            print(f"电脑出了{computer}，你输了")


def main2():
    global Y_1,Y_2,W_1,W_2,B_1,B_2
    count,win_count,fail_count=0,0,0
    while True:

        if count==10000 :break

        user=random_guess()
        if user==0:
            Y=np.array([[0,1,0]]).T
        elif user==1:
            Y=np.array([[0,0,1]]).T
        elif user==2:
            Y=np.array([[1,0,0]]).T

        computer = intell_guess()#猜拳

        #print(Y_2)

        #梯度下降
        dZ_2=Y_2-Y
        dW_2=np.dot((dZ_2),Y_1.T)
        dB_2=dZ_2
        dZ_1=np.dot(W_2.T,dZ_2)*(Y_1*(1-Y_1))
        dW_1=np.dot((dZ_1),X.T)
        dB_1=dZ_1

        W_2-=lr*dW_2
        W_1-=lr*dW_1
        B_2-=lr*dB_2
        B_1-=lr*dB_1

        count+=1
        if (user == 0 and computer == 2) or (user == 1 and computer == 0) or (user == 2 and computer == 1):
            print(f"电脑出了{computer}，你赢了")
            fail_count+=1
        elif computer == user:
            print(f"电脑出了{computer}，平局")
        else:
            print(f"电脑出了{computer}，你输了")
            win_count+=1

    print(f"电脑胜率：{round(win_count/count*100,2)}%")
    print(f"电脑败率：{round(fail_count/count*100,2)}%")



if __name__ == "__main__":
    main2()