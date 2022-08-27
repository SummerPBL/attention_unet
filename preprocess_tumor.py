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

def calculate_frame(raw_arr:np.ndarray)->Tuple[int]:
    """
    return (up,down,left,right)
    """
    assert(len(raw_arr.shape)==2)
    max_down=raw_arr.shape[0]-1

    up=0
    for up in range(0,max_down+1):
        if(np.any(raw_arr[up])==True):
            break
    
    down=max_down
    for down in reversed(range(0,max_down+1)):
        if(np.any(raw_arr[down])==True):
            break

    raw_arr_T=raw_arr.T
    max_down=raw_arr_T.shape[0]-1
    left=0
    for left in range(0,max_down+1):
        if(np.any(raw_arr_T[left])==True):
            break
    
    right=max_down
    for right in reversed(range(0,max_down+1)):
        if(np.any(raw_arr_T[right])==True):
            break

    return (up,down,left,right,)

def center_square_zoom(raw_img:np.ndarray,edge_len:int)->np.ndarray:
    """
    将任意形状的图片raw_img, 置于正方形中央并放大; 
    
    edge_len:正方形边长
    """
    assert(len(raw_img.shape)==2)
    side_len=max(raw_img.shape[0],raw_img.shape[1])
    board:np.ndarray=np.zeros(shape=(side_len,side_len,))
    up=(side_len-raw_img.shape[0])//2
    left=(side_len-raw_img.shape[1])//2
    # print('中心扩充',raw_img.shape,board.shape,up,left)
    board[up:up+raw_img.shape[0],left:left+raw_img.shape[1]]=raw_img
    
    PADDING=10
    board=cv2.resize(board,(edge_len-2*PADDING,edge_len-2*PADDING,))

    final=np.zeros(shape=(edge_len,edge_len,))
    final[PADDING:PADDING+board.shape[0],\
        PADDING:PADDING+board.shape[1]]=board
    return final


def crop_square_zoom(raw_img:np.ndarray,side_len:int,\
    label_img:Optional[np.ndarray]=None)->Tuple[np.ndarray]:
    """
    论文中的：放大、居中预处理
    返回2张图片
    """
    assert(len(raw_img.shape)==2)

    up,down,left,right=calculate_frame(raw_img)

    print(up,down,left,right)
    
    region_of_interest:np.ndarray=raw_img[up:down+1,left:right+1]
    # print(region_of_interest.shape)

    trimmed_img=center_square_zoom(region_of_interest,side_len,)
    print('预处理后图片大小:',trimmed_img.shape)

    if label_img is None:
        return trimmed_img
    
    region_of_interest=label_img[up:down+1,left:right+1]
    trimmed_label=center_square_zoom(region_of_interest,side_len,)
    # label取值应当只有0,1
    trimmed_label[trimmed_label<0.5]=0
    trimmed_label[trimmed_label>=0.5]=1
    return (trimmed_img,trimmed_label,)


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

    upper = 200
    lower = -200
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


    # 只考虑肿瘤
    WHITE_VALUE :np.int32 =255
    origin_label:np.ndarray=copy.deepcopy(Label_array)
    Label_array[Label_array > 0.5] = 1

    THRESHOLD :float =0.1
        
    for i in range(Label_array.shape[0]):
        label_select:np.ndarray = Label_array[i]
        if label_select.sum()/(256*256) < THRESHOLD:
            continue
        # print('某张2D:',label_select.shape) 256x256

        # label_arr: 肝脏背景
        # original_label: 肝脏 肿瘤 背景
        # tumor_label: 肿瘤 背景
        tumor_label=origin_label[i]
        tumor_label[tumor_label<1.5]=0
        tumor_label[tumor_label>=1.5]=1


        # img: 肝脏之外的部分为0
        # raw_pic=Image.fromarray(CT_array[i]*label_select).convert('L')
        # raw_pic=crop_square_zoom(CT_array[i]*label_select,256).convert('L')
        
        raw_arr:np.ndarray
        label_arr:np.ndarray
        raw_arr,label_arr=crop_square_zoom(CT_array[i]*label_select,256,tumor_label)
        label_arr*=WHITE_VALUE
        
        print('Image数据结构:',raw_arr.shape,label_arr.shape)
        if (label_arr.sum()>tumor_label.sum()):
            print('label也扩大了')
        elif(tumor_label.sum()==0):
            print('label不含肿瘤')
        else:
            exit(-1)

        raw_savepath=os.path.join(output_dir,'{}_{}.png'.format(NII_ID,i))
        raw_pic=Image.fromarray(raw_arr).convert('L')
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
    
    for i in range(47,131):
        if(len(os.listdir('./tumor_dataset/train'))>=500*2):
            break
        GetImFromNII(\
            f'D:/microsoft_PBL/3DUNet-Pytorch/raw_dataset/train/ct/volume-{i}.nii',\
            f'D:/microsoft_PBL/3DUNet-Pytorch/raw_dataset/train/label/segmentation-{i}.nii',\
            './tumor_dataset/train')
    
    print('done')

if __name__ == '__main2__':
    GetImFromNII('E:/3DUNet-Pytorch/raw_dataset/train/ct/volume-14.nii','E:/3DUNet-Pytorch/raw_dataset/train/label/segmentation-14.nii','./dataset')
    
    for i in range(40,131):
        if (len(os.listdir('./dataset/val'))>=50*2):
            break
        GetImFromNII(f'E:/3DUNet-Pytorch/raw_dataset/train/ct/volume-{i}.nii',f'E:/3DUNet-Pytorch/raw_dataset/train/label/segmentation-{i}.nii','./dataset/val')
    print('done')
