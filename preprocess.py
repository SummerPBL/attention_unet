from typing import Tuple,Optional
import SimpleITK as sitk
from PIL import Image


import numpy as np
import cv2

import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import os
from scipy import ndimage
import copy

def containOnly01(raw_arr:np.ndarray):
    tmp=raw_arr.flatten()
    for x in tmp:
        if x!=0 and x!=1:
            return False
    return True

def containOnly012(raw_arr:np.ndarray):
    tmp=raw_arr.flatten()
    for x in tmp:
        if x!=0 and x!=1 and x!=2:
            return False
    return True


def GetImFromNII(NII_impath:str, NII_segpath:str,output_dir:str)->None:
    NII_ID:int
    left,right=NII_impath.rfind('-'),NII_impath.rfind('.')
    NII_ID=int(NII_impath[left+1:right])
    print('nii文件编号',NII_ID)

    img = sitk.ReadImage(NII_impath)
    CT_array:np.ndarray = sitk.GetArrayFromImage(img)
    print('图片数据结构:',type(CT_array),CT_array.shape)

    seg = sitk.ReadImage(NII_segpath)
    Label_array:np.ndarray = sitk.GetArrayFromImage(seg)

    upper = 400
    lower = -100
    CT_array[CT_array > upper] = upper
    CT_array[CT_array < lower] = lower
    
    print('get sapcing',img.GetSpacing()[-1],seg.GetSpacing())
    slice_down_scale = 1
    xy_down_scale = 0.5
    CT_array = ndimage.zoom(CT_array, (img.GetSpacing()[-1] / slice_down_scale, xy_down_scale, xy_down_scale), order=3)
    
    Label_array = ndimage.zoom(Label_array, (seg.GetSpacing()[-1] / slice_down_scale, xy_down_scale, xy_down_scale), order=0)

    # 效果：变窄一半, 切片数量不变
    print('zoom后',CT_array.shape,Label_array.shape)


    CT_array=CT_array.astype(np.float32)
    np.round_(Label_array)

    # 检查label arr是否只有0,1,2三种取值 √
    # if containOnly01(Label_array[361]):
    #     print('label只有0,1两种取值')
    # elif containOnly012(Label_array[361]):
    #     print('label只有0,1,2,三种取值')
    # else:
    #     raise RuntimeError('label取值不止0,1,2')


    # 只考虑肝脏
    WHITE_VALUE :np.int32 =255
    origin_label:np.ndarray=copy.deepcopy(Label_array)
    Label_array[Label_array > 0.5] = 1

    THRESHOLD :float =0.1

    # 像素自适应 图像增强
    clahe = cv2.createCLAHE(clipLimit = 4, tileGridSize=(10,5))
        
    for i in range(Label_array.shape[0]):
        label_select:np.ndarray = Label_array[i]
        if label_select.sum()/(256*256) < THRESHOLD:
            continue
        # print('某张2D:',label_select.shape) 256x256

        # label_arr: 肝脏背景, 只含01
        # original_label: 肝脏 肿瘤 背景 只含012
        # tumor_label: 肿瘤 背景, 只含02
        tumor_label:np.ndarray=origin_label[i]
        tumor_label[tumor_label<1.5]=0
        tumor_label[tumor_label>=1.5]=1
        if tumor_label.sum()==0:
            continue

        # img: 肝脏之外的部分为0
        # raw_pic=Image.fromarray(CT_array[i]*label_select).convert('L')
        # raw_pic=crop_square_zoom(CT_array[i]*label_select,256).convert('L')
        
        label_arr:np.ndarray=label_select
        raw_pic=Image.fromarray(CT_array[i]).convert('L')
        raw_pic_arr=np.uint8(raw_pic)
        pic_clahe:np.ndarray=clahe.apply(raw_pic_arr)
        label_arr*=WHITE_VALUE
        

        raw_savepath=os.path.join(output_dir,'{}_{}.png'.format(NII_ID,i))
        raw_pic=Image.fromarray(pic_clahe).convert('L')
        print('---图像另存为---',raw_pic)
        raw_pic.save(raw_savepath)
        
        label_savepath=os.path.join(output_dir,'{}_{}_mask.png'.format(NII_ID,i))
        label_pic=Image.fromarray(label_arr).convert('1')
        label_pic.save(label_savepath)
        


if __name__ == '__main__':
    # GetImFromNII(\
    #     'D:/microsoft_PBL/3DUNet-Pytorch/raw_dataset/train/ct/volume-14.nii',\
    #     'D:/microsoft_PBL/3DUNet-Pytorch/raw_dataset/train/label/segmentation-14.nii',\
    #     './tmp_dataset')
    
    for i in range(27,47):
        if(len(os.listdir('./tmp_test'))>=600*2):
            break
        GetImFromNII(\
            f'D:/microsoft_PBL/3DUNet-Pytorch/raw_dataset/test/ct/volume-{i}.nii',\
            f'D:/microsoft_PBL/3DUNet-Pytorch/raw_dataset/test/label/segmentation-{i}.nii',\
            './tmp_test')
    
    # for i in range(47,130):
    #     if(len(os.listdir('./tmp_dataset'))>=2000*2):
    #         break
    #     GetImFromNII(\
    #         f'D:/microsoft_PBL/3DUNet-Pytorch/raw_dataset/train/ct/volume-{i}.nii',\
    #         f'D:/microsoft_PBL/3DUNet-Pytorch/raw_dataset/train/label/segmentation-{i}.nii',\
    #         './tmp_dataset')
    
    print('done')

if __name__ == '__main2__':
    GetImFromNII('E:/3DUNet-Pytorch/raw_dataset/train/ct/volume-14.nii','E:/3DUNet-Pytorch/raw_dataset/train/label/segmentation-14.nii','./dataset')
    
    for i in range(40,131):
        if (len(os.listdir('./dataset/val'))>=50*2):
            break
        GetImFromNII(f'E:/3DUNet-Pytorch/raw_dataset/train/ct/volume-{i}.nii',f'E:/3DUNet-Pytorch/raw_dataset/train/label/segmentation-{i}.nii','./dataset/val')
    print('done')
