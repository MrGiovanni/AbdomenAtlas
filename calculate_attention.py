import os
import argparse
import numpy as np
import nibabel as nib
from utils.utils import ORGAN_NAME_OVERLAP,TEMPLATE,ORGAN_NAME_LOW
from utils.utils import entropy_post_process,std_post_process,get_key
from tqdm import tqdm
import csv

def write_attention_size(args):
    dataset_name_list = [1,2,3]
    organ_target = TEMPLATE['target']
    attention_size = []
    case_name = []
    z_size = []
    for item in args.dataset_list:
        with open(os.path.join(args.data_txt_path,item + '.txt'), "r") as fl:
            all_lines = fl.readlines()
        for line in tqdm(range(len(all_lines))):
            dataset_name = all_lines[line].strip().split()[0].split('/')[0]
            if int(dataset_name[0:2]) == 10:
                template_key = get_key(all_lines[line].strip().split()[0].split('.')[0])
            else: 
                template_key = get_key(dataset_name)
            task_folder = os.path.join(args.data_root_path,dataset_name)
            current_case_name = all_lines[line].split()[0].split('.')[0].split('/')[-1]
            case_name.append(current_case_name)
            total_attention_size = 0
            organ_dataset = TEMPLATE[template_key]
            organ_index = [organ for organ in organ_target if organ not in organ_dataset]
            print('calculate attention map size for %s'%(current_case_name))

            if int(dataset_name[0:2]) in dataset_name_list:
                uncertainty_path = os.path.join(task_folder,current_case_name,'average','uncertainty')
                overlap_path = os.path.join(task_folder,current_case_name,'average','overlap')
                for idx in tqdm(range(len(organ_index))):
                    organ_name = ORGAN_NAME_LOW[organ_index[idx]-1]
                    uncertainty = nib.load(os.path.join(uncertainty_path,organ_name+'.nii.gz')).get_fdata()
                    overlap = nib.load(os.path.join(overlap_path,organ_name+'.nii.gz')).get_fdata()
                    attention = uncertainty/255+overlap
                    attention = attention/np.max(attention)
                    attention[np.isnan(attention)] = 0
                    total_attention_size += np.sum(attention)
            else: 
                attention_path = os.path.join(task_folder,current_case_name,'average','attention')
                for idx in tqdm(range(len(organ_index))):
                    organ_name = ORGAN_NAME_LOW[organ_index[idx]-1]
                    attention = nib.load(os.path.join(attention_path,organ_name+'.nii.gz')).get_fdata()
                    attention = attention/255
                    attention = attention/np.max(attention)
                    attention[np.isnan(attention)] = 0
                    total_attention_size += np.sum(attention)
            attention_size.append(total_attention_size)
            z_size.append(attention.shape[2])
   
    normalize_attention = np.array(attention_size)/np.array(z_size)
    attention_size_sorted_idx = np.argsort(-normalize_attention)
    print('sort case complete')
    if os.path.isfile(os.path.join(args.csv_save_path,dataset_name,args.csv_name+'.csv')):
        os.remove(os.path.join(args.csv_save_path,dataset_name,args.csv_name+'.csv'))
    for index in attention_size_sorted_idx:
        row = [case_name[index],normalize_attention[index]]
        with open(os.path.join(args.csv_save_path,dataset_name,args.csv_name+'.csv'),'a',newline='') as f:
            writer = csv.writer(f,delimiter=',', quotechar='"')
            writer.writerow(row)
  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT_123457891213', 'PAOT_10_inner']) # 'PAOT', 'felix'
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')
    parser.add_argument('--data_root_path', default='/ccvl/net/ccvl15/chongyu/LargePseudoDataset/', help='data root path')
    parser.add_argument('--csv_save_path',default = '/ccvl/net/ccvl15/chongyu/LargePseudoDataset/', help='csv save path')
    parser.add_argument('--csv_name',default = 'attention_size',help='attention size csv file name')
    args = parser.parse_args()

    write_attention_size(args)
if __name__ == "__main__":
    main()