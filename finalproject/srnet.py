import torch.nn as nn

#SRCNN  
#ref: 
# Dong C, Loy C C, He K, et al. Learning a deep convolutional network for image super-resolution[C]
# //European conference on computer vision. Springer, Cham, 2014: 184-199.


class SRCNN(nn.Module):
    def __init__(self,test=False):
        super(SRCNN,self).__init__()
          

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            #nn.BatchNorm2d(16),
            nn.ReLU())
            #nn.Dropout(0.1),  # drop 10% of the neuron
            #nn.ReLU(),
            #nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, padding=0),
            #nn.BatchNorm2d(16),
            nn.ReLU())
            #nn.Dropout(0.1),  # drop 10% of the neuron
            #nn.ReLU(),
            #nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=5, padding=2))
            #nn.BatchNorm2d(16))
            #nn.Dropout(0.2),  # drop 50% of the neuron
            #nn.ReLU(),
            #nn.MaxPool2d(2))

        if test: #测试情形
            for param in self.parameters():
                param.requires_grad = False#测试时不进行梯度下降，防止显存爆炸


    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)


        return output


