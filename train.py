from typing import Tuple
import torch
from torch.utils.data import DataLoader

from dataset import liverDataset
from attention_unet import attention_Unet
from dice_loss import binary_dice_loss
import numpy as np
import os


CONFIG_device:torch.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG_weights_save_path:str='./weights'

BATCH_SIZE:np.int32=4

model=attention_Unet(1,1)

bce_loss_func=torch.nn.BCELoss()

optimizer=torch.optim.Adam(model.parameters())

liver_dataset=liverDataset('./dataset',None,None)

data_loader=DataLoader(liver_dataset,BATCH_SIZE,shuffle=True,num_workers=0)

def train_iteration(model:attention_Unet,optimizer:torch.optim.Adam,raw_imgs:torch.Tensor,labels:torch.Tensor)->Tuple[float]:
    if model.training == False:
        model.train()

    optimizer.zero_grad()
    # forward
    output:torch.Tensor=model(raw_imgs)

    bce:torch.Tensor=bce_loss_func(output,labels)
    dice:torch.Tensor=binary_dice_loss(output,labels)

    total_loss=(bce+dice)/2
    # backward & update
    total_loss.backward()
    optimizer.step()

    return (bce.item(), dice.item(), total_loss.item(),)

# def validate(model:attention_Unet, optimizer:torch.optim.Adam,raw_img:torch.Tensor, label:torch.Tensor):
#     assert(len(raw_img.shape)==3 or len(raw_img.shape)==4)
#     if raw_img.shape==3:



if __name__=='__main__':
    model=model.to(CONFIG_device)
    model.train()
    
    print(type(optimizer))
    print(type(data_loader))
    img_num=len(liver_dataset)
    for epoch in range(20):
        bce_loss, dice_loss, total_loss=0.0, 0.0, 0.0
        print('------epoch{}------'.format(epoch))
        for i,(raw_imgs,labels) in enumerate(data_loader):
            raw_imgs:torch.Tensor
            labels:torch.Tensor
            raw_imgs,labels = raw_imgs.to(CONFIG_device),labels.to(CONFIG_device)

            bce,dice,total=train_iteration(model,optimizer,raw_imgs,labels)
            bce_loss+=bce
            dice_loss+=dice
            total_loss+=total
            if i%5==0:
                print('loss-- bce: {}, dice: {}, total: {}'.format(bce,dice,total))
                torch.save(model.state_dict(),os.path.join(CONFIG_weights_save_path,'attention_unet_{}_{}_{}'.format(epoch,i,total)))
        print('-------epoch ends {} {} {}--------'.format(bce_loss/img_num,dice_loss/img_num,total_loss/img_num))
        

