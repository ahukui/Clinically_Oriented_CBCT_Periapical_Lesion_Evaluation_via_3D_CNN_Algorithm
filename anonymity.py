"""
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
#Qwer@1234
import sys
#sys.path.append('D:/Software/Python/Install/Lib/site-packages')


import pydicom as dicom
import os
import numpy as np
#import cv2

folder_path = "D:\BaiduNetdiskDownload\Ori"   # Specify the .dcm folder path
out_folder_path = "D:\BaiduNetdiskDownload\OriT"  # Specify the output jpg/png folder path

folderlist = os.listdir(folder_path)

total_num_folder = len(folderlist)
print("This check totally have %d folders" % (total_num_folder))

new_folder_list = []
for n in folderlist:
    new_folder_list.append((n))
folderlist = sorted(new_folder_list)
folderlist= list(map(str,folderlist))

print(folderlist)


#os.mkdir('D:/segmentation_python/NAI_DATA/dicom-read/A001')

i=1
for sub_folder in folderlist:
    k=1
    sub_path = os.path.join(folder_path, sub_folder)
    print(sub_path)
    sub_folder_list=os.listdir(sub_path)
    print("NO. %s patient have %d sub_folders." % (sub_folder, len(sub_folder_list)))

    sub_output_file_name=out_folder_path + str(sub_folder)
    if os.path.exists(sub_output_file_name) is False:
        os.mkdir(sub_output_file_name)
    sub_output_path=os.path.join(out_folder_path, sub_folder+ "\\")
    print(sub_output_path, sub_path)
    ID = 0
    for folder in sub_folder_list:
            print(folder)
           # if 'RTSTRUCT' in folder:
            ds = dicom.dcmread(os.path.join(sub_path,folder),force=True)
           # print(ds)
               # cv2.imwrite(sub_output_path+folder+'.png',ds.pixel_array)
           # np.save(sub_output_path+str(ID)+'.npy',ds.pixel_array)
            ds.PatientName = ""
            ds.PatientID = "A_" + str(sub_folder)
            ds.PatientBirthDate = ""
            ds.PatientSex = ""
            ds.PatientAge = ""
            ds.PatientWeight = ""
            ds.PatientAddress = ""
            ds.InstitutionName = ""
            ds.SOPInstanceUID = ""
            ds.Manufacturer = ""
            ds.StationName = ""
            ds.XRayTubeCurrent = ""
            #print(sub_output_path, folder)
            #dicom.w
            ds.save_as(sub_output_path+str(folder))
#D:\segmentation-python\NAI-DATA\dicom-read\A001
#            m = m + 1
#        k = k + 1
#
#    print("NO. %d folder patient name is:_ %s _ and the ID is: _%s" % (i, ds.PatientName,ds.PatientID))
#
#    i = i + 1
