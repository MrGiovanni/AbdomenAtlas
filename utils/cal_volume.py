# Calculate the volume for eight organs based on pseudo labels
import numpy as np
import nibabel as nib
import os
from tqdm import tqdm
import csv

organs = ['spleen','liver','kidney','stomach','gall_bladder','pancreas','aorta','postcava']
files = ['01_Multi-Atlas_Labeling','02_TCIA_Pancreas-CT','03_CHAOS','04_LiTS','05_KiTS','07_WORD','08_AbdomenCT-1K','09_AMOS',
        '10_Decathlon','12_CT-ORG','13_AbdomenCT-12organ']
path = '/mnt/zzhou82/LargePseudoDataset/'

for i in range(len(files)):
    print(files[i])
    record = [['file_name','spleen','liver','kidney','stomach','gall_bladder','pancreas','aorta','postcava']]
    images_file = os.listdir(path+files[i])
    if 'anomaly.csv' in images_file:
        images_file.remove('anomaly.csv')
    ct_image = nib.load(path+files[i]+'/'+images_file[0]+'/'+'ct.nii.gz')
    voxel_dims = ct_image.header.get_zooms()
    voxel_size = voxel_dims[0]*voxel_dims[1]*voxel_dims[2]
    for j in tqdm(range(len(images_file))):
        temp = [files[i]]
        mask = path+files[i]+'/'+images_file[j]+'/segmentations/'
        for organ in organs:
            if organ == 'kidney':
                organ_mask1 = mask+organ+'_left.nii.gz'
                organ_image1 = nib.load(organ_mask1)
                organ_data1 = organ_image1.get_fdata()
                organ_mask2 = mask+organ+'_right.nii.gz'
                organ_image2 = nib.load(organ_mask2)
                organ_data2 = organ_image2.get_fdata()
                organ_volume = (np.sum(organ_data1)+np.sum(organ_data2))*voxel_size / 1000
            else:
                organ_mask = mask+organ+'.nii.gz'
                organ_image = nib.load(organ_mask)
                organ_data = organ_image.get_fdata()
                organ_volume = np.sum(organ_data)*voxel_size / 1000
            temp.append(organ_volume)
        record.append(temp)
    filename = '/data2/tzhang/LargePseudoDataset/Volume_CSV/'+files[i]+'.csv'
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(record)
