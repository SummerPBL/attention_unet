from typing import Tuple
import torch
from torch.utils.data import DataLoader

from dataset import liverDataset
from attention_unet import attention_Unet
from dice_loss import binary_dice_loss,binary_dice_coeff
import numpy as np
import os
import platform


CONFIG_device:torch.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG_weights_save_path:str='./weights'

CONFIG_num_workers = 0 if platform.system()=='Windows' else 4

BATCH_SIZE:np.int32=4

model=attention_Unet(1,1)

bce_loss_func=torch.nn.BCELoss()

optimizer=torch.optim.Adam(model.parameters())

train_dataset=liverDataset('./dataset/train',None,None)

train_loader=DataLoader(train_dataset,BATCH_SIZE,shuffle=True,num_workers=CONFIG_num_workers)

val_dataset=liverDataset('./dataset/val',None,None)

val_loader=DataLoader(val_dataset,BATCH_SIZE,shuffle=True,num_workers=CONFIG_num_workers)

def train_iteration(model:attention_Unet,optimizer:torch.optim.Adam,raw_imgs:torch.Tensor,labels:torch.Tensor)->Tuple[float]:
    if model.training == False:
        model.train()

    optimizer.zero_grad()
    # forward
    output:torch.Tensor=model(raw_imgs)

    bce:torch.Tensor=bce_loss_func(output,labels)
    dice:torch.Tensor=binary_dice_loss(output,labels)

    total_loss= bce+dice
    # backward & update
    total_loss.backward()
    optimizer.step()

    return (bce.item(), dice.item(), total_loss.item(),)

def validate(model:attention_Unet, data_loader:DataLoader):
    if model.training==True:
        model.eval()
    
    score=0
    total_count=len(data_loader.dataset)
    print(total_count)
    with torch.no_grad():

        for i,(raw_imgs,labels) in enumerate(data_loader):
            print(i,end=' ')
            raw_imgs:torch.Tensor
            labels:torch.Tensor
            raw_imgs, labels=raw_imgs.to(CONFIG_device),labels.to(CONFIG_device)

            pred=model(raw_imgs)

            dice_grade=binary_dice_coeff(pred,labels)

            score+=dice_grade.item()*labels.size(0)
    
    print()
    return score/total_count

            
if __name__=='__main__':
    model=model.to(CONFIG_device)
    model.train()
    
    print(type(optimizer))
    print(type(train_loader))
    img_num=len(train_dataset)
    for epoch in range(20):
        bce_loss, dice_loss, total_loss=0.0, 0.0, 0.0
        print('------epoch{}------'.format(epoch))
        for i,(raw_imgs,labels) in enumerate(train_loader):
            print(i,end=' ')
            if i>=10:
                break
            raw_imgs:torch.Tensor
            labels:torch.Tensor
            raw_imgs,labels = raw_imgs.to(CONFIG_device),labels.to(CONFIG_device)

            bce,dice,total=train_iteration(model,optimizer,raw_imgs,labels)
            bce_loss+=bce
            dice_loss+=dice
            total_loss+=total
            if i%100==0:
                print('loss-- bce: {}, dice: {}, total: {}'.format(bce,dice,total))
        print()
        print('-------train epoch ends {} {} {}--------'.format(bce_loss/img_num,dice_loss/img_num,total_loss/img_num))
        print('======eval======')
        dice_score:float=validate(model,val_loader)
        print('dice score: ',dice_score)
        torch.save(model.state_dict(),os.path.join(CONFIG_weights_save_path,'attention_unet_{}_{:d}.pth'.format(epoch,int(dice_score*1000))))
        

