import torch

from attention_unet import attention_Unet
from dice_loss import binary_dice_coeff

import numpy as np

from PIL import Image
import torchvision

import matplotlib.pyplot as plt

import os

def effect_show(rawimg_path:str,label_path:str,model:attention_Unet)->float:
    img=Image.open(rawimg_path).convert('L')
    label=Image.open(label_path).convert('1')

    trans_func=torchvision.transforms.ToTensor()
    img_arr, label_arr=trans_func(img), trans_func(label)

    input=img_arr.unsqueeze(0)
    # print(img_arr.size(),input.shape)
    if model.training==True:
        model.eval()
    with torch.no_grad():
        pred:torch.Tensor = model(input)
    pred.round_()

    plt.subplot(1,3,1)
    plt.imshow(torch.squeeze(img_arr).numpy())
    plt.title('original image')

    plt.subplot(1,3,2)
    plt.imshow(torch.squeeze(label_arr).numpy())
    plt.title('ground truth')

    plt.subplot(1,3,3)
    plt.imshow(torch.squeeze(pred).detach().numpy())
    plt.title('prediction')

    plt.show()

    dice_grade:torch.Tensor =binary_dice_coeff(pred,label_arr.unsqueeze(0))
    return dice_grade.item()

if __name__ == '__main__':
    model=attention_Unet(1,1,)
    # model.load_state_dict(torch.load('D:/microsoft_PBL/attention_unet/weights/attention_unet_10_10_0.04192398488521576',map_location='cpu'))

    model.load_state_dict(torch.load('D:/microsoft_PBL/attention_unet/weights/attention_unet_16_0_0.023037731647491455',map_location='cpu'))

    model.eval()
    print(model.training)

    grade=effect_show('D:/microsoft_PBL/attention_unet/dataset/14_329.png','D:/microsoft_PBL/attention_unet/dataset/14_329_mask.png',model)

    print('dice score:%.3f'%grade)