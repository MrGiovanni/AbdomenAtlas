import os
import argparse
import numpy as np
import nibabel as nib
from utils.utils import ORGAN_NAME_OVERLAP,TEMPLATE,ORGAN_NAME_LOW,MERGE_MAPPING_v2
from utils.utils import get_key,calculate_dice,surface_dice
from tqdm import tqdm
from scipy import ndimage
import csv

def dice_nsd(args):
    for item in args.dataset_list:
        organ_index = TEMPLATE['target']
        if not os.path.isdir(args.csv_save_path):
            os.makedirs(args.csv_save_path)
        if args.dice:
            if os.path.isfile(os.path.join(args.csv_save_path,args.dice_csv_name+'.csv')):
                os.remove(os.path.join(args.csv_save_path,args.dice_csv_name+'.csv'))
            row_dice = ['case id','spleen','kidney_right','kidney_left','gall_bladder','liver','stomach','aorta','postcava','pancreas']
            with open(os.path.join(args.csv_save_path,args.dice_csv_name+'.csv'),'a',newline='') as df:
                writer = csv.writer(df,delimiter=',', quotechar='"')
                writer.writerow(row_dice)
        if args.nsd:
            if os.path.isfile(os.path.join(args.csv_save_path,args.nsd_csv_name+'.csv')):
                os.remove(os.path.join(args.csv_save_path,args.nsd_csv_name+'.csv'))
            row_nsd = ['case id','spleen','kidney_right','kidney_left','gall_bladder','liver','stomach','aorta','postcava','pancreas']
            with open(os.path.join(args.csv_save_path,args.nsd_csv_name+'.csv'),'a',newline='') as nf:
                writer = csv.writer(nf,delimiter=',', quotechar='"')
                writer.writerow(row_nsd)
        with open(os.path.join(args.data_txt_path,item + '.txt'), "r") as fl:
            all_lines = fl.readlines()
        for line in tqdm(range(len(all_lines))):
            if args.internal:
                dataset_name = all_lines[line].strip().split()[0].split('/')[0]
                CT_volume_name = all_lines[line].strip().split()[0].split('/')[0]
                gt_path = os.path.join(args.data_root_path,CT_volume_name,'original_label.nii.gz')
                seg_path = os.path.join(args.data_root_path,CT_volume_name,'segmentations')
            else: 
                dataset_name = all_lines[line].strip().split()[0].split('/')[0]
                CT_volume_name = all_lines[line].strip().split()[0].split('.')[0].split('/')[-1]
                gt_path = os.path.join(args.data_root_path,dataset_name,CT_volume_name,'original_label.nii.gz')
                seg_path = os.path.join(args.data_root_path,dataset_name,CT_volume_name,'backbones','swinunetr','segmentations')
            gt_data = nib.load(gt_path).get_fdata()
            gt_header = nib.load(gt_path).header
            spacing = [dim.astype(float) for dim in gt_header['pixdim'][1:4]]
            row_dice = [CT_volume_name]
            row_nsd = [CT_volume_name]
            for idx in tqdm(range(len(organ_index))):
                organ_name = ORGAN_NAME_LOW[organ_index[idx]-1]
                organ = nib.load(os.path.join(seg_path,organ_name+'.nii.gz')).get_fdata()
                gt_organ = np.zeros(gt_data.shape)
                task_key = get_key(dataset_name)
                task_gt_mapping = MERGE_MAPPING_v2[task_key]
                for mapping_key in task_gt_mapping:
                    organ_universal_index,organ_gt_index = mapping_key
                    if organ_index[idx] == organ_universal_index:
                        gt_organ[gt_data==organ_gt_index]=1 
                if np.sum(gt_organ) != 0:
                    if args.dice:
                        organ_dice = calculate_dice(organ,gt_organ)
                    if args.nsd:
                        organ_nsd = surface_dice(organ,gt_organ,spacing,1)
                else:
                    organ_dice = np.NaN
                    organ_nsd = np.NaN
                if args.dice:
                    row_dice.append(organ_dice)
                if args.nsd:
                    row_nsd.append(organ_nsd)
            if args.dice:
                with open(os.path.join(args.csv_save_path,args.dice_csv_name+'.csv'),'a',newline='') as df:
                    writer = csv.writer(df,delimiter=',', quotechar='"')
                    writer.writerow(row_dice)
            if args.nsd:
                with open(os.path.join(args.csv_save_path,args.nsd_csv_name+'.csv'),'a',newline='') as nf:
                    writer = csv.writer(nf,delimiter=',', quotechar='"')
                    writer.writerow(row_nsd)




            


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_path', default='/ccvl/net/ccvl15/chongyu/LargePseudoDataset/', help='The data root path')
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT_123457891213', 'PAOT_10_inner']) # 'PAOT', 'felix'
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')
    parser.add_argument('--csv_save_path', default='/ccvl/net/ccvl15/chongyu/Dice', help='The csv save path')
    parser.add_argument('--internal', action="store_true", default=False, help='whether use internal dataset')
    parser.add_argument('--dice', action="store_true", default=False, help='whether calculate dice')
    parser.add_argument('--nsd', action="store_true", default=False, help='whether calculate nsd')
    parser.add_argument('--dice_csv_name', default='dice', help='dice csv file name')
    parser.add_argument('--nsd_csv_name', default='nsd', help='nsd csv file name')
    args = parser.parse_args()
    
    dice_nsd(args)

if __name__ == "__main__":
    main()