import os
import argparse
import torch
import numpy as np
import nibabel as nib
from utils.utils import ORGAN_NAME_OVERLAP,TEMPLATE,ORGAN_NAME_LOW,MERGE_MAPPING_v2
from utils.utils import get_key,calculate_metrics,create_entropy_map
from tqdm import tqdm
from scipy import ndimage
import csv

def nnunet_post_processing(args):
    organ_index = TEMPLATE['01'] # modify if not use BTCV for nnunet training
    target_organ_index = TEMPLATE['target']
    for item in args.dataset_list:
        for line in open(args.data_txt_path + item + '_test.txt'):
            # task_folder = save_dir +'/'+ line.strip().split()[0].split('/')[0]
            task_name = line.strip().split()[0].split('/')[0]
            current_case_name = line.strip().split()[0].split('.')[0].split('/')[-1]
            print('post processing for %s'%(current_case_name))
            current_case_mask_path = os.path.join(args.data_root_path,current_case_name+'.nii.gz')
            current_case_npz_path = os.path.join(args.data_root_path,current_case_name+'.npz')
            current_case_seg_save_path = os.path.join(args.save_dir,task_name,current_case_name,'backbones','nnunet','segmentations')
            current_case_soft_pred_save_path = os.path.join(args.save_dir,task_name,current_case_name,'backbones','nnunet','soft_pred')
            current_case_entropy_save_path = os.path.join(args.save_dir,task_name,current_case_name,'backbones','nnunet','entropy')
            
            if not os.path.isdir(current_case_seg_save_path):
                os.makedirs(current_case_seg_save_path)
            if not os.path.isdir(current_case_soft_pred_save_path):
                os.makedirs(current_case_soft_pred_save_path)
            if not os.path.isdir(current_case_entropy_save_path):
                os.makedirs(current_case_entropy_save_path)
            
            current_case_mask_load = nib.load(current_case_mask_path)
            affine_temp = current_case_mask_load.affine
            current_case_mask_data = current_case_mask_load.get_fdata()
            current_case_npz_load = np.load(current_case_npz_path)
            current_case_npz_data = current_case_npz_load[current_case_npz_load.files[0]]
            for idx in organ_index:
                # store organ seg
                organ_name = ORGAN_NAME_LOW[idx-1]
                organ_mask = np.zeros(current_case_mask_data.shape)
                organ_mask[current_case_mask_data == idx] = 1
                organ_mask_save = nib.Nifti1Image(organ_mask.astype(np.uint8),affine_temp)
                nib.save(organ_mask_save,os.path.join(current_case_seg_save_path,organ_name+'.nii.gz'))
                print('success save %s segmentation'%(organ_name))

                # store organ soft pred and entropy
                # soft_pred
                organ_softmax = current_case_npz_data[idx]
                organ_softmax = organ_softmax.transpose(2,1,0)

                struct2 = ndimage.generate_binary_structure(3, 3)
                organ_mask_dilation = ndimage.binary_dilation(organ_mask,structure=struct2,iterations=1)

                organ_softmax[organ_mask_dilation==0] = 0
                organ_softmax_save = nib.Nifti1Image(organ_softmax.astype(np.uint8),affine_temp)
                nib.save(organ_softmax_save,os.path.join(current_case_soft_pred_save_path,organ_name+'.nii.gz'))
                print('success save %s soft prediction'%(organ_name))

                # entropy
            for target_idx in target_organ_index:
                target_organ_name = ORGAN_NAME_LOW[target_idx-1]
                organ_softmax_for_entropy = current_case_npz_data[target_idx].transpose(2,1,0)
                organ_softmax_for_entropy = torch.from_numpy(organ_softmax_for_entropy/255)
                organ_entropy = torch.special.entr(organ_softmax_for_entropy).numpy()*255
                organ_entropy_save = nib.Nifti1Image(organ_entropy.astype(np.uint8),affine_temp)
                nib.save(organ_entropy_save,os.path.join(current_case_entropy_save_path,target_organ_name+'.nii.gz'))
                print('success save %s entropy'%(target_organ_name))

                



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT_123457891213', 'PAOT_10_inner']) # 'PAOT', 'felix'
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')
    parser.add_argument('--data_root_path', default='/dev/shm/tiezheng/nnUNET', help='data root path')
    parser.add_argument('--save_dir', default='/dev/shm/chongyu/LargePesudoDataset', help='data root path')
    args = parser.parse_args()

    nnunet_post_processing(args)

if __name__ == "__main__":
    main()