from typing import List, Tuple,Optional
import torch
from torch.utils.data import DataLoader
# from torch.utils.tensorboard.writer import SummaryWriter
import time

from dataset import liverDataset
import nested_unet
import encoding_unetpp
from dice_loss import binary_dice_loss,binary_dice_coeff,weightedBCE
import numpy as np
import os
import platform
from pathlib import Path
from multiprocessing import cpu_count
import toolkit


# configuration
CONFIG_DEVICE:torch.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ENCODER_LOAD_PATH='./trained_models/encoder_18_level4_9798.pth'

suffix:str=time.strftime('%m-%d+%H-%M-%S', time.localtime(time.time()))
WEIGHTS_SAVE_DIR:str='./weights_net++'
WEIGHTS_SAVE_DIR+=suffix
if Path(WEIGHTS_SAVE_DIR).is_dir()==False:
    os.mkdir(WEIGHTS_SAVE_DIR)

CONFIG_NUM_WORKERS = 0 if platform.system()=='Windows' else min(max(cpu_count()-2,0),10)

BATCH_SIZE:np.int32=2

USE_BOTTLE_NECK=True

DEBUG_MODE:bool=True

print('-----------configuration-----------')
print('Device:',CONFIG_DEVICE)
print('Workers number:',CONFIG_NUM_WORKERS)
print('-----------------------------------')

# Plotting
LOG_DIR='./log_common'

LOG_DIR+=suffix
if Path(LOG_DIR).is_dir()==False:
    os.mkdir(LOG_DIR)
print('statistics:',LOG_DIR)
SAMPLE_NUM_EPOCH = 3

# neural networks
model=nested_unet.NestedUNet(1,1,)
ref_model = encoding_unetpp.NestedUNet(1,1,) if USE_BOTTLE_NECK else None
if ref_model is not None:
    ref_model.load_state_dict(\
        torch.load(ENCODER_LOAD_PATH, map_location='cpu'))
    ref_model.to(CONFIG_DEVICE)
    ref_model.eval()
    print('reference encoder loads successfully √')


# loss functions
L1_loss_func=torch.nn.SmoothL1Loss().to(CONFIG_DEVICE)

optimizer=torch.optim.Adam(model.parameters())

train_dataset=liverDataset('./dataset/train',None,None)

train_loader=DataLoader(train_dataset,BATCH_SIZE,shuffle=True,num_workers=CONFIG_NUM_WORKERS)

val_dataset=liverDataset('./dataset/val',None,None)

val_loader=DataLoader(val_dataset,BATCH_SIZE,shuffle=False,num_workers=CONFIG_NUM_WORKERS)

# exit()

def train_iteration(model:nested_unet.NestedUNet, \
        ref_model:Optional[encoding_unetpp.NestedUNet], \
        optimizer:torch.optim.Adam, \
        raw_imgs:torch.Tensor,labels:torch.Tensor)->Tuple[float]:
    """
    return float(bce, dice, maeL1, total_loss,)
    forward + backward + update on raw_imgs
    """
    if model.training == False:
        model.train()

    optimizer.zero_grad()
    # forward
    x1_0,x2_0,x3_0,x4_0,x0_1,x0_2,x0_3,x0_4=model.multi_forward(raw_imgs)

    # calculate loss
    outputs:Tuple[torch.Tensor]=(x0_1,x0_2,x0_3,x0_4,)
    bce_loss:torch.Tensor=0
    dice_loss:torch.Tensor=0
    for prediction in outputs:
        bce_loss+=weightedBCE(prediction,labels,0.8)
        dice_loss+=binary_dice_loss(prediction,labels)

    total_loss:torch.Tensor= bce_loss+dice_loss

    if ref_model!=None:
        if ref_model.training == True:
            ref_model.eval()
        with torch.no_grad():
            ref_x1_0,ref_x2_0,ref_x3_0,ref_x4_0=ref_model.encode(raw_imgs)
        L1_loss:torch.Tensor = L1_loss_func(x1_0,ref_x1_0) \
            +L1_loss_func(x2_0,ref_x2_0) \
            +L1_loss_func(x3_0,ref_x3_0) \
            +L1_loss_func(x4_0,ref_x4_0)
        total_loss+=L1_loss
    else:
        L1_loss=torch.zeros(1)

    # backward & update
    total_loss.backward()
    optimizer.step()

    return (bce_loss.item()/4, dice_loss.item()/4, L1_loss.item()/4, total_loss.item()/4,)

def validate(model:nested_unet.NestedUNet, data_loader:DataLoader)->Tuple[float]:
    """
    return float(score1,2,3,4)
    """
    if model.training==True:
        model.eval()
    
    score1,score2,score3,score4=0.0, 0.0, 0.0, 0.0
    total_count:int=0
    print('<----validate /{}---->'.format(len(data_loader)))
    with torch.no_grad():
        for i,(raw_imgs,labels) in enumerate(data_loader):            
            raw_imgs:torch.Tensor
            labels:torch.Tensor
            raw_imgs, labels=raw_imgs.to(CONFIG_DEVICE),labels.to(CONFIG_DEVICE)

            x0_1:torch.Tensor
            x0_2:torch.Tensor
            x0_3:torch.Tensor
            x0_4:torch.Tensor
            x0_1,x0_2,x0_3,x0_4=model.multi_predict(raw_imgs)
            x0_1.round_()
            x0_2.round_()
            x0_3.round_()
            x0_4.round_()

            dice_grade1=binary_dice_coeff(x0_1,labels,eval_mode=True)
            dice_grade2=binary_dice_coeff(x0_2,labels,eval_mode=True)
            dice_grade3=binary_dice_coeff(x0_3,labels,eval_mode=True)
            dice_grade4=binary_dice_coeff(x0_4,labels,eval_mode=True)

            score1+=dice_grade1.item()*labels.size(0)
            score2+=dice_grade2.item()*labels.size(0)
            score3+=dice_grade3.item()*labels.size(0)
            score4+=dice_grade4.item()*labels.size(0)
            total_count+=labels.size(0)

            if DEBUG_MODE==True:
                assert(dice_grade1.item()>=0 and dice_grade1.item()<=1)
                assert(dice_grade2.item()>=0 and dice_grade2.item()<=1)
                assert(dice_grade3.item()>=0 and dice_grade3.item()<=1)
                assert(dice_grade4.item()>=0 and dice_grade4.item()<=1)
                print('check reasonal dice score √')
                break
    
    return score1/total_count,score2/total_count,score3/total_count,score4/total_count,

            
if __name__=='__main__':
    model=model.to(CONFIG_DEVICE)
    model.train()
    
    print(type(optimizer))
    print(type(train_loader))

    modulus:int=int(np.ceil(len(train_loader)/SAMPLE_NUM_EPOCH))

    # Statistics
    bce_loss_batches:List[float]=[]
    dice_loss_batches:List[float]=[]
    smoothL1_loss_batches:List[float]=[]

    bce_loss_epochs:List[float]=[]
    dice_loss_epochs:List[float]=[]
    smoothL1_loss_epochs:List[float]=[]

    dice_score_epochs:List[List[float]] =[]
    

    for epoch in range(20):
        if epoch>=20:
            ref_model=None
        bce_loss, dice_loss, L1_loss, total_loss=0.0, 0.0, 0.0, 0.0
        total_count:int=0
        print('------epoch{}------'.format(epoch))
        print('<======Train, total batches: {}======>'.format(len(train_loader)))
        for i,(raw_imgs,labels) in enumerate(train_loader):
            raw_imgs:torch.Tensor
            labels:torch.Tensor
            raw_imgs,labels = raw_imgs.to(CONFIG_DEVICE),labels.to(CONFIG_DEVICE)

            bce, dice, mae, total= \
                train_iteration(model,ref_model,optimizer,raw_imgs,labels)
            
            bce_loss+=bce*labels.size(0)
            dice_loss+=dice*labels.size(0)
            L1_loss+=mae*labels.size(0)
            total_loss+=total*labels.size(0)
            total_count+=labels.size(0)
            if i%modulus==0:
                print('\tProgress: {}/{}| loss: bce={}, dice={}, smoothL1={}, total={}' \
                    .format(i,len(train_loader), bce,dice,mae, total))
                bce_loss_batches.append(bce)
                dice_loss_batches.append(dice)
                smoothL1_loss_batches.append(mae)
            if(DEBUG_MODE==True):
                break
        print()
        print('-------Train done, loss: bce={}, dice={}, smoothL1={}, total={},--------'\
            .format(bce_loss/total_count,dice_loss/total_count, \
                    L1_loss/total_count,total_loss/total_count))
        bce_loss_epochs.append(bce_loss/total_count)
        dice_loss_epochs.append(dice_loss/total_count)
        smoothL1_loss_epochs.append(L1_loss/total_count)
        print('<======eval======>')
        dice_score1,dice_score2,dice_score3,dice_score4 \
            = validate(model,val_loader)
        dice_arr=(dice_score1,dice_score2,dice_score3,dice_score4,)
        dice_score_epochs.append(dice_arr)
        print('dice score(1~4): ',dice_arr)
        best_level=np.argmax(dice_arr)

        torch.save(model.state_dict(),os.path.join(WEIGHTS_SAVE_DIR,'unet++_{}_level{}_{:04d}.pth'.format(epoch,best_level+1,int(dice_arr[best_level]*10000))))

        if DEBUG_MODE==True:
            break
        
    """
    Plot the statistics
    """
    dice_score_epochs_:np.ndarray=np.array(dice_score_epochs)
    dice_score_epochs_=dice_score_epochs_.T
    assert(dice_score_epochs_.shape[0]==4)

    if USE_BOTTLE_NECK==False:
        smoothL1_loss_epochs=None
        smoothL1_loss_batches=None
    
    toolkit.log_statistics(LOG_DIR,SAMPLE_NUM_EPOCH,\
        dice_loss_batches,bce_loss_batches,smoothL1_loss_batches,\
        dice_loss_epochs,bce_loss_epochs,smoothL1_loss_epochs,dice_score_epochs_)