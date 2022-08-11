from typing import Tuple,Optional
import torch
from torch.utils.data import DataLoader

from dataset import liverDataset
import third_unet
from dice_loss import binary_dice_loss,binary_dice_coeff
import numpy as np
import os
import platform
from pathlib import Path
from multiprocessing import cpu_count


# configuration
CONFIG_DEVICE:torch.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WEIGHTS_SAVE_DIR:str='./third_weights_net++'
if Path(WEIGHTS_SAVE_DIR).is_dir()==False:
    os.mkdir(WEIGHTS_SAVE_DIR)

CONFIG_NUM_WORKERS = 0 if platform.system()=='Windows' else min(max(cpu_count()-2,0), 10) 

BATCH_SIZE:np.int32=2

DEBUG_MODE:bool=True

USE_DEEP_SUPERVISION=True

print('-----------configuration-----------')
print('Device:',CONFIG_DEVICE)
print('Workers number:',CONFIG_NUM_WORKERS)
print('-----------------------------------')

# neural networks
model=third_unet.NestedUNet(1,1,USE_DEEP_SUPERVISION)

# loss functions
bce_loss_func=torch.nn.BCELoss().to(CONFIG_DEVICE)
mse_loss_func=torch.nn.MSELoss().to(CONFIG_DEVICE)

optimizer=torch.optim.Adam(model.parameters())

train_dataset=liverDataset('./dataset/train',None,None)

train_loader=DataLoader(train_dataset,BATCH_SIZE,shuffle=True,num_workers=CONFIG_NUM_WORKERS)

val_dataset=liverDataset('./dataset/val',None,None)

val_loader=DataLoader(val_dataset,BATCH_SIZE,shuffle=True,num_workers=CONFIG_NUM_WORKERS)

# exit()

def train_iteration(model:third_unet.NestedUNet, \
        optimizer:torch.optim.Adam, \
        raw_imgs:torch.Tensor,labels:torch.Tensor)->Tuple[float]:
    """
    return (bce, dice, total_loss,)
    forward + backward + update on raw_imgs
    """
    if model.training == False:
        model.train()

    optimizer.zero_grad()
    # forward
    x0_1:torch.Tensor
    x0_2:torch.Tensor
    x0_3:torch.Tensor
    x0_4:torch.Tensor
    bce_loss:torch.Tensor
    dice_loss:torch.Tensor
    if USE_DEEP_SUPERVISION==True:
        x0_1,x0_2,x0_3,x0_4=model.forward(raw_imgs)
        bce_loss=0
        dice_loss=0
        for x in (x0_1,x0_2,x0_3,x0_4,):
            bce_loss+=bce_loss_func(x,labels)
            dice_loss+=binary_dice_loss(x,labels)
        bce_loss/=4
        dice_loss/=4
    else:
        x0_4=model.forward(raw_imgs)
        bce_loss=bce_loss_func(x0_4,labels)
        dice_loss=binary_dice_loss(x0_4,labels)


    total_loss:torch.Tensor= bce_loss+dice_loss

    # backward & update
    total_loss.backward()
    optimizer.step()

    return (bce_loss.item(), dice_loss.item(), total_loss.item(),)

def validate(model:third_unet.NestedUNet, data_loader:DataLoader)->Tuple[float]:
    if model.training==True:
        model.eval()
    
    score1,score2,score3,score4=0.0, 0.0, 0.0, 0.0
    total_count=len(data_loader.dataset)
    print('<----validate /{}---->'.format(len(data_loader)))
    with torch.no_grad():
        for i,(raw_imgs,labels) in enumerate(data_loader):
            print(i,end=' ')
            
            raw_imgs:torch.Tensor
            labels:torch.Tensor
            raw_imgs, labels=raw_imgs.to(CONFIG_DEVICE),labels.to(CONFIG_DEVICE)

            x0_1:torch.Tensor
            x0_2:torch.Tensor
            x0_3:torch.Tensor
            x0_4:torch.Tensor
            if USE_DEEP_SUPERVISION:
                x0_1,x0_2,x0_3,x0_4=model(raw_imgs)
                dice_grade1=binary_dice_coeff(x0_1,labels)
                dice_grade2=binary_dice_coeff(x0_2,labels)
                dice_grade3=binary_dice_coeff(x0_3,labels)
                dice_grade4=binary_dice_coeff(x0_4,labels)
            else:
                x0_4=model(raw_imgs)
                dice_grade1,dice_grade2,dice_grade3=\
                    torch.zeros(1),torch.zeros(1),torch.zeros(1)
                dice_grade4=binary_dice_coeff(x0_4,labels)
                

            score1+=dice_grade1.item()*labels.size(0)
            score2+=dice_grade2.item()*labels.size(0)
            score3+=dice_grade3.item()*labels.size(0)
            score4+=dice_grade4.item()*labels.size(0)

            if DEBUG_MODE==True:
                assert(dice_grade1.item()>=0 and dice_grade1.item()<=1)
                assert(dice_grade2.item()>=0 and dice_grade2.item()<=1)
                assert(dice_grade3.item()>=0 and dice_grade3.item()<=1)
                assert(dice_grade4.item()>=0 and dice_grade4.item()<=1)
                print('check reasonal dice score(0<x<1) √')
                break
    
    print()
    return score1/total_count,score2/total_count,score3/total_count,score4/total_count,

            
if __name__=='__main__':
    model=model.to(CONFIG_DEVICE)
    model.train()

    
    print(type(optimizer))
    print(type(train_loader))
    img_num=len(train_dataset)
    for epoch in range(20):
        bce_loss, dice_loss, total_loss=0.0, 0.0, 0.0
        print('------epoch{}------'.format(epoch))
        print('<----train /{}---->'.format(len(train_loader)))
        for i,(raw_imgs,labels) in enumerate(train_loader):
            print(i,end=' ')

            raw_imgs:torch.Tensor
            labels:torch.Tensor
            raw_imgs,labels = raw_imgs.to(CONFIG_DEVICE),labels.to(CONFIG_DEVICE)

            bce, dice, total= \
                train_iteration(model,optimizer,raw_imgs,labels)
            bce_loss+=bce*labels.size(0)
            dice_loss+=dice*labels.size(0)
            total_loss+=total*labels.size(0)
            if i%100==0:
                print('single batch loss-- bce: {}, dice: {}, total: {}'.format(bce,dice, total))
            if(DEBUG_MODE==True):
                break
        print()
        print('-------train loss: {}, {}, {},--------'.format(bce_loss/img_num,dice_loss/img_num,total_loss/img_num))
        print('======eval======')
        dice_score1,dice_score2,dice_score3,dice_score4 \
            = validate(model,val_loader)
        dice_arr=(dice_score1,dice_score2,dice_score3,dice_score4,)
        print('dice score(1~4): ',dice_arr)
        best_level=np.argmax(dice_arr)

        torch.save(model.state_dict(),os.path.join(WEIGHTS_SAVE_DIR,'third_unet++_{}_level{}_{:d}.pth'.format(epoch,best_level+1,int(dice_arr[best_level]*10000))))

        if DEBUG_MODE==True:
            break
        