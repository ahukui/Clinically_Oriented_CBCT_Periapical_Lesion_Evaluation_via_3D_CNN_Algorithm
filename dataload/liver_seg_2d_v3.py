import numpy as np
import cv2
import glob 
import os 
from torch.utils.data import Dataset
import torch 
import imgaug as ia 
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imgaug.augmenters as iaa 
import random
import SimpleITK as sitk 
import json
from torch.nn.functional import interpolate
from scipy import ndimage
import time
import copy
def write_nii(image_array, save_path):
    image_array = image_array.astype(np.uint8)
    image = sitk.GetImageFromArray(image_array)
    sitk.WriteImage(image, save_path)

def load_nii(nii_path):
    image = sitk.ReadImage(nii_path)
    image_array = sitk.GetArrayFromImage(image)
    return image_array

class npyDataSet(Dataset):
    def __init__(self, fold, num_image=64, transform_indx = 0,  if_test=False, windowCenter = 450, windowWidth = 2000):
        data_path = './fold_list/'
        #self.label_paths = ann["label_path"]
        self.transform_indx = transform_indx
        #self.pred_paths = ann["pred_path"]
        self.if_test = if_test
        self.num_image = num_image
        self.windowCenter = windowCenter
        self.windowWidth = windowWidth  
        self.image_paths = []
        for i in range(5):
            print(i,fold)
            ann = json.load(open(data_path+'%d.json'%i))
            if i != fold:
                print(len(ann["label_path"]))
                self.image_paths.extend(ann["label_path"])
                print('~~~~~~~~num',len(self.image_paths))            
            # else:
            #     self.image_paths = ann["label_path"]       
            #     print('~~~~~~~~num',len(self.image_paths))          
        
        # ia.seed(2)
        self.seq = iaa.Sequential([
            iaa.Affine(rotate=(-30, 30), shear=(-20, 20), translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}),
            iaa.PerspectiveTransform(scale=(0., 0.1)),
            iaa.PiecewiseAffine(scale=(0, 0.025)),
            # iaa.ElasticTransformation(alpha=(0, 10.0), sigma=(0, 1)),
            # iaa.AdditiveGaussianNoise(scale=(0, 5), per_channel=True),
            iaa.AdditiveLaplaceNoise(scale=(0, 0.05*255), per_channel=True),
            iaa.AdditivePoissonNoise((0, 10), per_channel=True),
            # iaa.Dropout(p=(0, 0.1), per_channel=0.5),
            iaa.ImpulseNoise((0,0.01)),
            iaa.GammaContrast((0.75, 1.5)),
            iaa.Crop(px=(0, 32)),
        ])
        self.cnt = 0

    def __len__(self):
        return len(self.image_paths)
    
    def get_bbox_from_mask(self, Orimask, outside_value=0):
        mask = copy.copy(Orimask)
        mask[mask<1]=0
        mask[mask>0]=1
        mask_voxel_coords = np.where(mask != outside_value)
        minzidx = max(int(np.min(mask_voxel_coords[0])), 0)
        maxzidx = min(int(np.max(mask_voxel_coords[0]))+1, mask.shape[0])
        minxidx = int(np.min(mask_voxel_coords[1]))
        maxxidx = int(np.max(mask_voxel_coords[1])) + 1
        minyidx = int(np.min(mask_voxel_coords[2]))
        maxyidx = int(np.max(mask_voxel_coords[2])) + 1
        return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


    def transform(self, image, label):
        image = np.transpose(image, (1,2,0))
        h, w = label.shape[1:]
        label = np.transpose(label, (1,2,0)).reshape((1, h, w, -1))
        image_aug, label_aug = self.seq(image=image, segmentation_maps=label)
        label_aug = label_aug[0]
        label_aug = np.transpose(label_aug, (2,0,1))
        image_aug = np.transpose(image_aug, (2,0,1))
        return image_aug, label_aug


    def get_center_and_enlage(self, OriSize, minIdx, maxIdx, AimSize):

        center_idx = int((minIdx + maxIdx)/2)
        NminIdx = max(0, round(center_idx - AimSize / 2))
        NmaxIdx = min(OriSize, round(center_idx + AimSize / 2))
        if (NmaxIdx - NminIdx) < AimSize:
            if NminIdx == 0:
                NmaxIdx = NminIdx + AimSize
            if NmaxIdx == OriSize:
                NminIdx = NmaxIdx - AimSize
        
        return NminIdx, NmaxIdx

    def crop_patch(self, image, label, crop_size):

        num_slices, H, W = image.shape
        [c, h, w] = crop_size
        start_index = random.randint(0, num_slices - c)
        start_x = random.randint(0, H - h)
        start_y = random.randint(0, W - w)

        _label = label[start_index : start_index + c, start_x :start_x + h, start_y :start_y + w]
        _image = image[start_index : start_index + c, start_x :start_x + h, start_y :start_y + w]
        #print(np.sum(_label))
        return _image, _label


    def get_tumor_region(self, label):
        mask = ndimage.label(label>1)[0]
        ind_list = [] 
        num_slices, H, W = label.shape
        #print(mask.max())
        for _index in range(1, mask.max()+1):
            z, y, x = np.where(mask==_index)
            ind_list.append([int((z.min() + z.max())/2), int((y.min() + y.max())/2), int((x.min()+x.max())/2)])
        return ind_list


    def normalized(self, image):
        image = image.astype('float')
        minWindow = float(self.windowCenter) - 0.5 * float(self.windowWidth)
        maxWindow = float(self.windowCenter) + 0.5 * float(self.windowWidth)
        image[image < minWindow] = minWindow
        image[image > maxWindow] = maxWindow
        mean = np.mean(image)
        std = np.std(image)
        image -= mean
        image /= std        
        return image 


    def normalized2(self, image):
        #image = image.astype('float')
        minWindow = -600
        maxWindow = 1400
        image[image < minWindow] = minWindow
        image[image > maxWindow] = maxWindow
        image = (image - minWindow)/ (maxWindow - minWindow)
        image = 2 * image - 1
        return image 

    def __getitem__(self, idx):
        t1 = time.time()
        label_path = self.image_paths[idx]
        label = load_nii(label_path)
        #print(image_path)
        fileName = os.path.split(label_path)[1]
        image_path = label_path.replace('AllLabel','AllData').replace('Label.nii','IM.nii')
        image = load_nii(image_path)

        t2 = time.time()
        #print('load time~~~~~~~~~',t2-t1)
        t1 = time.time()
        if label.max()<1:
            return self.__getitem__(random.randint(0, len(self.image_paths)-1))  
        else:

            C_size, H_size, W_size = 32, 256, 256#image.shape[1:]    
            [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]] = self.get_bbox_from_mask(label)
            #print(minzidx-maxzidx, minxidx-maxxidx, minyidx-maxyidx)
            if maxzidx - minzidx < C_size:
                minzidx, maxzidx = self.get_center_and_enlage(label.shape[0], minzidx, maxzidx, C_size)

            if maxxidx - minxidx < H_size:
                minxidx, maxxidx = self.get_center_and_enlage(label.shape[1], minxidx, maxxidx, H_size)
                
            if maxyidx - minyidx < W_size:
                minyidx, maxyidx = self.get_center_and_enlage(label.shape[2], minyidx, maxyidx, W_size)        
        
            _image = image[minzidx : maxzidx , minxidx : maxxidx, minyidx : maxyidx]
            _label = label[minzidx : maxzidx , minxidx : maxxidx, minyidx : maxyidx] 
            t2 = time.time()
            #print('crop time~~~~~~~~~',t2-t1)
            t1 = time.time()
            if _image.shape[0] < C_size:
                _image = np.concatenate((_image, _image), axis=0)
                _label = np.concatenate((_label, _label), axis=0)
        
            c, h, w = 32, 256, 256
            _image, _label = self.crop_patch(_image, _label, [c, h, w])  
            
            cnt = 0
            while _label.max()<1 and cnt<3:
                cnt += 1
                _image, _label = self.crop_patch(_image, _label, [c, h, w])  
                
                
            _index = random.randint(0, 1)
            if _index!=0:
                trans = [(0, 1, 2), (0, 2, 1)]
                _label = np.transpose(_label, trans[_index])
                _image = np.transpose(_image, trans[_index])

            _image, _label = torch.Tensor(_image.astype("float").copy()), torch.Tensor(_label.astype("uint8").copy())#.unsqueeze(0)   .unsqueeze(0)         
            #print(image.shape, label.shape)
            
            if random.random()>0.25:
                offset = random.randint(-10, 10)
                condi = (_label==1)
                _image[condi] = _image[condi] + (torch.rand_like(_image[condi])) * offset
                offset = random.randint(-30, 30)
                condi = (_label==0)
                _image[condi] = _image[condi] + (torch.rand_like(_image[condi])) * offset
                

            _image = self.normalized2(_image)
            return _image, _label, fileName
    
if __name__=="__main__":
    dataset = npyDataSet("./CBCT_Task/Code/data.json", 32)
    for index in range(100):
        image, label, fileName = dataset[index]
        print(image.shape, label.shape)
        #image = image/3*127+127
        #image = image.squeeze().numpy().astype(np.uint8)
        #label = label.squeeze().numpy().astype(np.uint8)
        #image = sitk.GetImageFromArray(image)
        #label = sitk.GetImageFromArray(label)
        #sitk.WriteImage(image, str(index).zfill(4)+"_0000.nii.gz")
        #sitk.WriteImage(label, str(index).zfill(4)+".nii.gz")
        #print(index)