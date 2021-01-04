import os
import cv2
import time
import glob
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from srnet import SRCNN
from PIL import Image, ImageFile
from function import RecurrentSampler
from alive_progress import alive_bar

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#训练设备
PATH = os.path.dirname(__file__)

####################options###################
parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('-L','--lr_dir',type=str, default="C:/train/DIV2K/DIV2K_train_LR_bicubic/X2",
                    help='Directory path to batchs of lr images')
parser.add_argument('-H','--hr_dir',type=str, default="C:/train/DIV2K/DIV2K_train_HR",
                    help='Directory path to batchs of hr images')

# Training options
parser.add_argument('-x','--x_zoom', type=int, default=2,
                    help='zoom scale')
parser.add_argument('--save_dir', type=str,default=os.path.join(PATH,'model/srnet'),
                    help='Directory to save the model')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('-i','--iter_times', type=int, default=10000)
parser.add_argument('-b','--batch_size', type=int, default=1)

parser.add_argument('--n_threads', type=int, default=0)


args = parser.parse_args()

hr_err=set()
lr_err=set()
tt=transforms.ToTensor()
up2=nn.Upsample(scale_factor=2, mode="bilinear")

#############定义训练数据集#############
class SRDataset(data.Dataset):
    '''数据集'''
    def __init__(self,lr_path,hr_path):
        super(SRDataset, self).__init__()

        self.lr_paths = glob.glob(lr_path+'/*.png')
        self.hr_paths = glob.glob(hr_path+'/*.png')

    def __getitem__(self, index):

        lr_path = self.lr_paths[index]
        hr_path = self.hr_paths[index]
        lr_img = Image.open(lr_path).convert('RGB')
        lr_img = tt(lr_img).to(device).float()
        hr_img = Image.open(hr_path).convert('RGB')
        hr_img = tt(hr_img).to(device).float()

        return lr_img,hr_img,lr_path,hr_path


    def __len__(self):
        return len(self.lr_paths)

    def name(self):
        return 'SRDataset'

def adjust_learning_rate(optimizer, iters):
    '''动态调整学习率'''
    lr = args.lr / (1.0 + args.lr_decay * iters)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(srcnn,srloader):
    '''训练解码器'''
    global lr_err,hr_err
    print('Start training...')
    tic=time.time()
    mseloss=nn.MSELoss()

    #使用adam优化器
    optimizer = torch.optim.Adam(srcnn.parameters(), lr=args.lr)
    #optimizer = optim.SGD(decoder.parameters(), lr=0.001, momentum=0.9) #优化函数为随机梯度下降

    with alive_bar(args.iter_times) as bar:
        for epoch in range(args.iter_times):
            bar()
            adjust_learning_rate(optimizer,epoch)
            lr_img,hr_img,lr_path,hr_path=next(srloader)
            lr_img=up2(lr_img)
            
            try:

                optimizer.zero_grad()
                output=srcnn(lr_img)
                loss=mseloss(output,hr_img)
                loss.backward()
                optimizer.step()
                del lr_img,hr_img,output
                
            except:
                print("error")
                torch.cuda.empty_cache()
                lr_err.add(lr_path)
                hr_err.add(hr_path)
                             
                continue

            print('iters: '+str(epoch)+'  loss:'+str(loss.item()))

        torch.save(srcnn.state_dict(), args.save_dir+f'_x{args.x_zoom}.pth')
        toc=time.time()

    print('TRIANING COMPLETED.')
    print('Time cost: {}s.'.format(toc-tic))
    print('model saved as:  '+args.save_dir+f'_x{args.x_zoom}.pth')
    print(lr_err)
    print(hr_err)


def main():

    #定义数据加载器

    srset = SRDataset(lr_path=args.lr_dir,hr_path=args.hr_dir)

    srloader = iter(data.DataLoader(srset, batch_size=args.batch_size, sampler=RecurrentSampler(srset),\
        num_workers=args.n_threads,pin_memory=False,drop_last=True))

    #初始化模型
    srcnn=SRCNN().to(device)
    srcnn.train()
    #训练网络
    train(srcnn,srloader)



if __name__ == "__main__":
    main()