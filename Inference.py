#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 15:27:18 2021

@author: zjxing
"""

import torch 
import glob 
import os 
# from model.vessel_2d_transformer import vesselnet as vesselnet_transformer_2d
from model.PALNet import vesselnet as PALNet
import numpy as np 
import SimpleITK as sitk 
from scipy.ndimage.filters import gaussian_filter
from torch.nn.functional import interpolate
import queue 
# from multiprocessing import Process,Queue
import multiprocessing as mp
import json
from scipy.ndimage import label
from skimage import measure
import copy
import time
from scipy import ndimage
from sklearn import metrics
import numpy as np
import cv2
import glob 
import os 
import random
import SimpleITK as sitk 
import json
from torch.nn.functional import interpolate
from scipy import ndimage

reader = sitk.ImageSeriesReader()

def write_nii(image_array, save_path):
    image_array = image_array.astype(np.uint8)
    image = sitk.GetImageFromArray(image_array)
    sitk.WriteImage(image, save_path)

def load_nii(nii_path):
    image = sitk.ReadImage(nii_path)
    Ori = image.GetOrigin()
    #print(image.GetSize())
    Space = image.GetSpacing()
    Direct = image.GetDirection()
    image_array = sitk.GetArrayFromImage(image)
    return image_array, Ori, Space, Direct

def load_dicom(image_path):
    image_name = reader.GetGDCMSeriesFileNames(image_path)
    reader.SetFileNames(image_name)
    image = reader.Execute()
    #space = image.GetSpacing()
    print(image.GetOrigin())
    print(image.GetSize())
    print(image.GetSpacing())
    print(image.GetDirection())
    image_array = sitk.GetArrayFromImage(image)
    return image_array, image.GetOrigin(), image.GetSpacing(), image.GetDirection()

def write_gz(gz_path, image_array, Ori, Space, Direct, mask= False):
    if mask:
        image_array = image_array.astype(np.uint8)
    image = sitk.GetImageFromArray(image_array)
    image.SetDirection(Direct)
    image.SetSpacing(Space)
    image.SetOrigin(Ori)
    sitk.WriteImage(image, gz_path)
    
    
    
def get_filelist(file, Filelist):
    newDir = file
    if os.path.isfile(file):
        Filelist.append(file)
    elif os.path.isdir(file):
        for s in os.listdir(file):
            newDir = os.path.join(file, s)
            get_filelist(newDir, Filelist)

    return Filelist

def findBiggestComponent(arr):
    seg = copy.copy(arr)
    seg[seg<1]=0
    seg[seg>0]=1
    seg = measure.label(seg, connectivity=2, background=0)  # 四联通法对seg标记
    vals, counts = np.unique(seg, return_counts=True)  # counts[]保存每个值在seg中出现的次数
    counts = counts[vals != 0]
    vals = vals[vals != 0]

    if len(counts) > 0:
        max_component = vals[np.argmax(counts)]
        seg = (seg == max_component)

    seg = seg.astype(np.uint8)
    return seg

def post_process_largest(mask):
    mask = label(mask)[0]
    max_num = 0
    max_cls_id = 0
    for cls_id in range(1, mask.max()+1):
        cls_num = mask[mask==cls_id].size
        if cls_num>max_num:
            max_num = cls_num
            max_cls_id = cls_id
    mask[mask!=max_cls_id] = 0
    return mask


def load_gz(gz_path):
    gz = sitk.ReadImage(gz_path)
    return sitk.GetArrayFromImage(gz)

def compute_dice(gt, mask):
    tp = gt[(mask==1)&(gt>0)].size
    fp = gt[(mask==1)&(gt==0)].size
    fn = gt[(mask!=1)&(gt>0)].size
    dice = (2*tp+1)/(tp*2+1+fp+fn)
    print('dice:', dice)
    return dice

def normalized_HF(image):
    image = image.astype('float')
    minWindow = -600
    maxWindow = 1400
    image[image < minWindow] = minWindow
    image[image > maxWindow] = maxWindow
    image = (image - minWindow)/ (maxWindow - minWindow)
    image = 2 * image - 1
    return image 

def remove_small_region(label_all):
        
        label = label_all.copy()
        if np.sum(label_all)>0:
            mask = ndimage.label(label_all>0)[0]
            for _index in range(1, mask.max()+1):
                x, y, z = np.where(mask==_index)
                #print(len(x))
                if len(x)<10:
                    label[x.min():x.max()+1,y.min():y.max()+1,z.min():z.max()+1]=0
                    # for i,j,k in zip(x,y,z):
                    #     label[i,j,k]=0
        return label
    

def test_model_by_3D_files_nii():

    ######### load model########
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = PALNet(num_seg_classes=1)
    model = torch.nn.DataParallel(model).cuda()
    pth = torch.load("./Weights/61000_stage.pth", map_location="cpu")["model"]
    model.load_state_dict(pth)
    model.cuda()    
    model.eval() 
    
    
    ################### get data ##########
    data_path = "./PartsData/"
    save_dir = "./SegRe/"#'/media/kui/zhuqikui/2023newCT-NiFTI-format/seg/'#'/media/kui/Dataset/PreNii/TestData/'#
    
    all_data = [name for name in os.listdir(data_path)]
    #print(all_data)
    for data in all_data:
        file_list = [name for name in os.listdir(os.path.join(data_path,data))]
        print(data)
        for file in file_list:
            if file.startswith('IM'):   
                print(file)
                Oriimage, Ori, Space, Direct = load_nii(os.path.join(data_path+data, file))#load_dicom(data_path+data)
                
                
        save_path= os.path.join(save_dir, data)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        save_path = os.path.join(save_dir, data)
        C_size, H_size, W_size = 32, 512, 512#image.shape[1:]    

        image = Oriimage
        image = normalized_HF(image)
        print(image.shape)
        
        mask = np.zeros_like(image)
        scores = np.zeros_like(image)
        weights = scores*0.+1e-6


        image = torch.Tensor(image)
        for slice_index in range(0, image.shape[0]+1, C_size//2):
                    if image.shape[0]-slice_index<C_size:
                        slice_index = image.shape[0] - C_size
                
            # for W_index in range(0, image.shape[1], W_size//2):
            #     if image.shape[1]-W_index<  W_size:
            #         W_index = image.shape[1] -  W_size               
                
            #     for H_index in range(0, image.shape[2], H_size//2):
            #         if image.shape[2]-H_index < H_size:
            #             H_index = image.shape[2] - H_size              
                
                    #print(slice_index, W_index, H_index, slice_index+C_size, W_index+W_size, H_index+H_size)
                    image_patch = image[slice_index:slice_index+C_size].cuda().unsqueeze(0)#, W_index:W_index+W_size, H_index:H_index+H_size]
                    #print('image_patch', image_patch.shape)
                    #image_patch = 3 * (image_patch - 127)/127
                    with torch.no_grad():
                        #image_patch = interpolate(image_patch[:, :, 32:480, 32:480], (512, 512), mode="bicubic", align_corners=True).unsqueeze(0)
                        image_patch = interpolate(image_patch, (512, 512), mode="bicubic", align_corners=True)#.cuda().unsqueeze(0)
                        #print(image_patch.shape)    
                        pred = model(image_patch.unsqueeze(1))[-1]                            
                        pred += model(image_patch.unsqueeze(1).flip(dims=(2,)))[-1].flip(dims=(2,))
                        pred += model(image_patch.unsqueeze(1).flip(dims=(3,)))[-1].flip(dims=(3,))     
                        pred += model(image_patch.unsqueeze(0).flip(dims=(4,)))[-1].flip(dims=(4,))
                        pred += model(image_patch.unsqueeze(0).flip(dims=(2, 3)))[-1].flip(dims=(2, 3))
                        pred += model(image_patch.unsqueeze(0).flip(dims=(2, 4)))[-1].flip(dims=(2, 4))   
                        pred += model(image_patch.unsqueeze(0).flip(dims=(3, 4)))[-1].flip(dims=(3, 4))
                        pred += model(image_patch.unsqueeze(0).flip(dims=(2, 3, 4)))[-1].flip(dims=(2, 3, 4))
                        pred /= 8.                        
                        
                        pred = pred.view(1, C_size, W_size, H_size)
                        #print(pred.shape)
                        pred = interpolate(pred, (image.shape[-2], image.shape[-1]), mode="bicubic", align_corners=True)                    
                        pred = pred.sigmoid().cpu().squeeze().numpy()
                        #print(type(pred), type(scores))
                        #print('~~~~~~~~', scores[slice_index:slice_index+C_size, W_index:W_index+W_size, H_index:H_index+H_size].shape, pred.shape)
                        scores[slice_index+1:slice_index+C_size-1] += pred[1:-1]#(pred.astype('float32')) #,W_index:W_index+W_size, H_index:H_index+H_size
                        weights[slice_index+1:slice_index+C_size-1] += np.ones_like(pred[1:-1])
                        

        scores = scores / weights
        mask[scores>=0.5] = 1
        mask[mask<1]=0
        print(mask.shape, np.max(mask))
        
        # if not os.path.exists(os.path.join(save_path,name)):
        #     os.makedirs(os.path.join(save_path,name))
        #Seg[minzidx : maxzidx , minxidx : maxxidx, minyidx : maxyidx] = mask[:maxzidx-minzidx,:,:]
        #Oriimage = crop_normalized(Oriimage)
        #write_gz(save_path+'/IM.nii', Oriimage, Ori, Space, Direct)
        #write_gz(save_path+'/Label.nii', Orilabel, Ori, Space, Direct, True)        
        write_gz(save_path+'/Seg.nii', mask, Ori, Space, Direct, True) 


if __name__=="__main__":
    #test_model_by_3D_new_files_nii()
    test_model_by_3D_files_nii()
    #volume_evaluate_tumor()
    #volume_evaluate_WBYZ()
    #volume_evaluate()


 
 
