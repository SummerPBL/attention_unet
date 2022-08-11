from typing import Tuple
import torch
from torch.utils.data import DataLoader

from dataset import liverDataset
import encoding_unetpp
from dice_loss import binary_dice_loss,binary_dice_coeff
import numpy as np
import os
import platform
from pathlib import Path
from multiprocessing import cpu_count


# configuration
CONFIG_DEVICE:torch.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WEIGHTS_SAVE_DIR:str='./weights_encoder'
if Path(WEIGHTS_SAVE_DIR).is_dir()==False:
    os.mkdir(WEIGHTS_SAVE_DIR)

CONFIG_NUM_WORKERS = 0 if platform.system()=='Windows' else min(max(cpu_count()-2,0),10)

BATCH_SIZE:np.int32=2

DEBUG_MODE:bool=True

print('-----------configuration-----------')
print('Device:',CONFIG_DEVICE)
print('Workers number:',CONFIG_NUM_WORKERS)
print('-----------------------------------')

# neural networks
model = encoding_unetpp.NestedUNet(1,1,)

# loss functions
bce_loss_func=torch.nn.BCELoss().to(CONFIG_DEVICE)
mse_loss_func=torch.nn.MSELoss().to(CONFIG_DEVICE)

optimizer=torch.optim.Adam(model.parameters())

train_dataset=liverDataset('./dataset/train',None,None)

train_loader=DataLoader(train_dataset,BATCH_SIZE,shuffle=True,num_workers=CONFIG_NUM_WORKERS)

val_dataset=liverDataset('./dataset/val',None,None)

val_loader=DataLoader(val_dataset,BATCH_SIZE,shuffle=True,num_workers=CONFIG_NUM_WORKERS)

# exit()

def train_iteration(model:encoding_unetpp.NestedUNet, \
        optimizer:torch.optim.Adam, \
        labels:torch.Tensor)->Tuple[float]:
    """
    return float(bce, dice, total_loss)
    forward + backward + update on labels
    """
    if model.training == False:
        model.train()

    optimizer.zero_grad()
    # forward
    x0_1, x0_2, x0_3, x0_4=model(labels)

    # calculate loss
    outputs:Tuple[torch.Tensor]=(x0_1,x0_2,x0_3,x0_4,)
    bce:torch.Tensor=0
    dice:torch.Tensor=0
    for prediction in outputs:
        bce+=bce_loss_func(prediction,labels)
        dice+=binary_dice_loss(prediction,labels)

    total_loss:torch.Tensor= bce+dice

    # backward & update
    total_loss.backward()
    optimizer.step()

    return (bce.item()/4, dice.item()/4, total_loss.item()/4,)

def validate(model:encoding_unetpp.NestedUNet, data_loader:DataLoader)->float:
    """
    return float(score4)
    """
    if model.training==True:
        model.eval()
    
    score4=0.0
    total_count:int=0
    print('<----validate /{}---->'.format(len(data_loader)))
    with torch.no_grad():
        for i,(_,labels) in enumerate(data_loader):
            print(i,end=' ')
            
            labels:torch.Tensor
            labels=labels.to(CONFIG_DEVICE)

            x0_4:torch.Tensor
            _,_,_,x0_4=model(labels)

            dice_grade4=binary_dice_coeff(x0_4,labels)

            score4+=dice_grade4.item()*labels.size(0)
            total_count+=labels.size(0)

            if DEBUG_MODE==True:
                assert(dice_grade4.item()>=0 and dice_grade4.item()<=1)
                print('check reasonal dice score √')
                break
    
    print()
    return score4/total_count

            
if __name__=='__main__':
    model=model.to(CONFIG_DEVICE)
    model.train()
    
    print(type(optimizer))
    print(type(train_loader))

    for epoch in range(20):
        bce_loss, dice_loss, total_loss=0.0, 0.0, 0.0
        total_count:int=0
        print('------epoch{}------'.format(epoch))
        print('<----train /{}---->'.format(len(train_loader)))
        for i,(_,labels) in enumerate(train_loader):
            print(i,end=' ')

            labels:torch.Tensor
            labels = labels.to(CONFIG_DEVICE)

            bce, dice, total= \
                train_iteration(model,optimizer,labels)
            
            bce_loss+=bce*labels.size(0)
            dice_loss+=dice*labels.size(0)
            total_loss+=total*labels.size(0)
            total_count+=labels.size(0)
            if i%100==0:
                print('loss-- bce: {}, dice: {}, total: {}'.format(bce,dice, total))
            if(DEBUG_MODE==True):
                break
        print()
        print('-------train loss: bce={}, dice={}, total={},--------'\
            .format(bce_loss/total_count,dice_loss/total_count, \
                    total_loss/total_count))
        print('======eval======')
        dice_score4 = validate(model,val_loader)
        print('4th dice score: ',dice_score4)

        torch.save(model.state_dict(),os.path.join(WEIGHTS_SAVE_DIR,'encoder_{}_level{}_{:d}.pth'.format(epoch,4,int(dice_score4*10000))))

        if DEBUG_MODE==True:
            break