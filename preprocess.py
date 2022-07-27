import SimpleITK as sitk
from PIL import Image


import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import os
from scipy import ndimage

def GetImFromNii(nii_impath:str, nii_segpath:str,output_dir:str)->None:
    nii_ID:int
    left,right=nii_impath.rfind('-'),nii_impath.rfind('.')
    nii_ID=int(nii_impath[left+1:right])
    print('nii文件编号',nii_ID)


    img = sitk.ReadImage(nii_impath)
    img_array = sitk.GetArrayFromImage(img)
    print(img_array.shape)

    seg = sitk.ReadImage(nii_segpath)
    seg_array:np.ndarray
    seg_array = sitk.GetArrayFromImage(seg)

    CT_array:np.ndarray
    CT_array = img_array[:,:,:]
    Label_array:np.ndarray
    Label_array = seg_array[:,:,:]
    # print(CT_array.shape,Label_array.shape)
    assert(CT_array.shape == Label_array.shape)

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
    # print('zoom后',CT_array.shape,Label_array.shape)

    CT_array=CT_array.astype(np.float32)
    np.round_(Label_array)

    # 只有0 1 2三种取值
    # tmp_arr=Label_array[361,:,:]
    # tmp_arr=tmp_arr.flatten()
    # for x in tmp_arr:
    #     if x!=0 and x!=1:
    #         print('不止01!!!!!!!!!!!!!!!!!',x)
            
    # print('01检查',tmp_arr.sum())


    # 将肝脏和肝肿瘤的标签融合为一个,做二分类
    WHITE_VALUE :np.int32 =255
    Label_array[Label_array > 0.5] = WHITE_VALUE

    # 只有0 1 2三种取值
    # tmp_arr=Label_array[361,:,:]
    # tmp_arr=tmp_arr.flatten()
    # for x in tmp_arr:
    #     if x!=0 and x!=WHITE_VALUE:
    #         print('不止01!!!!!!!!!!!!!!!!!',x)
            
    # print('01检查',tmp_arr.sum())

    THRESHOLD :float =0.1
        
    for i in range(Label_array.shape[0]):
        label_select = Label_array[i,:,:]
        if label_select.sum()/WHITE_VALUE/(256*256) < THRESHOLD:
            continue

        label_pic = Image.fromarray(label_select).convert('1')
        label_savepath=os.path.join(output_dir,'{}_{}_mask.png'.format(nii_ID,i))
        label_pic.save(label_savepath)

        raw_pic=Image.fromarray(CT_array[i,:,:]).convert('L')
        raw_savepath=os.path.join(output_dir,'{}_{}.png'.format(nii_ID,i))
        raw_pic.save(raw_savepath)

if __name__ == '__main__':
    GetImFromNii('E:/3DUNet-Pytorch/raw_dataset/train/ct/volume-14.nii','E:/3DUNet-Pytorch/raw_dataset/train/label/segmentation-14.nii','./dataset')
    
    for i in range(40,131):
        if (len(os.listdir('./dataset/val'))>=50*2):
            break
        GetImFromNii(f'E:/3DUNet-Pytorch/raw_dataset/train/ct/volume-{i}.nii',f'E:/3DUNet-Pytorch/raw_dataset/train/label/segmentation-{i}.nii','./dataset/val')
    print('done')
