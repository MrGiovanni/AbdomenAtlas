import os
import argparse
import torch
import numpy as np
import nibabel as nib
import shutil
from utils.utils import ORGAN_NAME_OVERLAP,TEMPLATE,ORGAN_NAME_LOW,MERGE_MAPPING_v2
from utils.utils import get_key,calculate_metrics,create_entropy_map,threshold_organ,organ_post_process
from utils.utils import pseudo_label_all_organ,pseudo_label_single_organ
from tqdm import tqdm
from scipy import ndimage
import csv

def average_and_create_dataset(args):
    organ_index = TEMPLATE['all']
    for item in args.dataset_list:
        for line in open(os.path.join(args.data_txt_path,item + '.txt')):
            dataset_name = line.strip().split()[0].split('/')[0]
            case_name = line.strip().split()[0].split('.')[0].split('/')[-1]
            ct_path = os.path.join(args.data_root_path,dataset_name,case_name,'ct.nii.gz')
            case_path = os.path.join(args.save_dir,dataset_name,case_name)
            pseudo_label_save_path = os.path.join(case_path,'average','pseudo_label.nii.gz')
            organ_seg_save_path = os.path.join(case_path,'average','segmentations')
            if not os.path.isdir(organ_seg_save_path):
                os.makedirs(organ_seg_save_path)
            if not os.path.isfile(os.path.join(case_path,'ct.nii.gz')):
                shutil.copy(ct_path, os.path.join(case_path,'ct.nii.gz'))


            ct_load = nib.load(ct_path)
            ct_data = ct_load.get_fdata()
            affine_temp = ct_load.affine
            W,H,D = ct_data.shape
            average_pred = np.zeros((1,32,W,H,D))
            count = np.zeros((1,32,W,H,D))
            count_list = np.zeros(32)
            for idx in organ_index:
                organ_name = ORGAN_NAME_LOW[idx-1]
                for model in args.model_list:
                    organ_path = os.path.join(args.data_root_path,dataset_name,case_name,'backbones',model,'soft_pred',organ_name+'.nii.gz')
                    if os.path.isfile(organ_path):
                        organ_data = nib.load(organ_path).get_fdata()
                        average_pred[0,idx-1] += organ_data/255
                        count[0,idx-1] += np.ones(organ_data.shape)
                        count_list[idx-1] += 1
            
            average_soft_pred = average_pred/count
            print('%s success average soft pred'%(case_name))
            print(average_soft_pred.shape)
            print(count_list)

            average_hard_pred = threshold_organ(torch.from_numpy(average_soft_pred),args)

            average_hard_pred_post,total_anomly_slice_number = organ_post_process(average_hard_pred.numpy(), organ_index,case_path,args)
            average_hard_pred_post = torch.tensor(average_hard_pred_post)

            
            for idx in organ_index:
                pseudo_label_single = pseudo_label_single_organ(average_hard_pred_post,idx,args)
                organ_name = ORGAN_NAME_LOW[idx-1]
                pseudo_label_single = pseudo_label_single.numpy().astype(np.uint8)
                organ_save = nib.Nifti1Image(np.squeeze(pseudo_label_single),affine_temp)
                print('organ seg saved in path: %s'%(os.path.join(organ_seg_save_path, organ_name+'.nii.gz')))
                nib.save(organ_save,os.path.join(organ_seg_save_path, organ_name+'.nii.gz'))
            
            pseudo_label_all = pseudo_label_all_organ(average_hard_pred_post,args)
            pseudo_label_all = pseudo_label_all.numpy().astype(np.uint8)
            pseudo_label_save = nib.Nifti1Image(np.squeeze(pseudo_label_all),affine_temp)
            nib.save(pseudo_label_save,pseudo_label_save_path)
            print('pseudo label saved in path: %s'%(pseudo_label_save_path))


            right_lung_data_path = os.path.join(organ_seg_save_path,'lung_right.nii.gz')
            left_lung_data_path = os.path.join(organ_seg_save_path,'lung_left.nii.gz')
            organ_name=['lung_right','lung_left']
            right_lung_data = nib.load(right_lung_data_path).get_fdata()
            left_lung_data = nib.load(left_lung_data_path).get_fdata()
            right_lung_data_sum = np.sum(right_lung_data,axis=(0,1))
            left_lung_data_sum = np.sum(left_lung_data,axis=(0,1))
            right_lung_size = np.sum(right_lung_data,axis=(0,1,2))
            left_lung_size = np.sum(left_lung_data,axis=(0,1,2))
            if right_lung_size != 0 or left_lung_size != 0:
                if right_lung_size>left_lung_size:
                    non_zero_idx = np.nonzero(right_lung_data_sum)
                    first_non_zero_idx = non_zero_idx[0][0]
                    if total_anomly_slice_number!=0:
                        for s in range(first_non_zero_idx,right_lung_data.shape[-1]):
                            if len(np.unique(ct_data[:,:,s]))>10  and right_lung_data_sum[s] ==0:
                                print('start writing csv at slice: '+str(s+1))
                                with open(os.path.join(args.save_dir,dataset_name,'average_anomaly.csv'),'a',newline='') as f:
                                    writer = csv.writer(f)
                                    content = case_name
                                    writer.writerow([content,organ_name[0],s+1])
                                    writer.writerow([content,organ_name[1],s+1])
                else: 
                    non_zero_idx = np.nonzero(left_lung_data_sum)
                    first_non_zero_idx = non_zero_idx[0][0]
                    if total_anomly_slice_number!=0:
                        for s in range(first_non_zero_idx,left_lung_data.shape[-1]):
                            if len(np.unique(ct_data[:,:,s]))>10  and left_lung_data_sum[s] ==0:
                                print('start writing csv at slice: '+str(s+1))
                                with open(os.path.join(args.save_dir,dataset_name,'average_anomaly.csv'),'a',newline='') as f:
                                    writer = csv.writer(f)
                                    content = case_name
                                    writer.writerow([content,organ_name[0],s+1])
                                    writer.writerow([content,organ_name[1],s+1])
            











def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT_123457891213', 'PAOT_10_inner']) # 'PAOT', 'felix'
    parser.add_argument('--model_list',nargs='+', default=['swinunetr', 'unet','nnunet'])
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='PAOT list path')
    parser.add_argument('--data_root_path', default='/ccvl/net/ccvl15/chongyu/LargePseudoDataset', help='data root path')
    parser.add_argument('--create_dataset',action="store_true", default=False, help='whether create atlas8k')
    parser.add_argument('--cpu',action="store_true", default=False, help='whether use cpu')
    parser.add_argument('--save_dir', default='/ccvl/net/ccvl15/chongyu/LargePseudoDataset', help='Atlas8k save path')
    args = parser.parse_args()

    average_and_create_dataset(args)

if __name__ == "__main__":
    main()
