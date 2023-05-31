import os
import argparse
import numpy as np
import nibabel as nib
from utils.utils import ORGAN_NAME_OVERLAP,TEMPLATE,ORGAN_NAME_LOW,ORGAN_NAME_OVERLAP
from utils.utils import entropy_post_process,std_post_process,get_key
from tqdm import tqdm
import shutil
import csv

def create_attention(args):
    organ_target = TEMPLATE['target']
    name_id = []
    attention_value = []
    sorted_name_id = []
    for item in args.dataset_list:
        with open(os.path.join(args.data_txt_path,item + '.txt'), "r") as f:
            all_lines = f.readlines()
        for line in tqdm(range(len(all_lines))):
            dataset_name = all_lines[line].strip().split()[0].split('/')[0]
            if int(dataset_name[0:2]) == 10:
                template_key = get_key(all_lines[line].strip().split()[0].split('.')[0])
            else: 
                template_key = get_key(dataset_name)
            organ_dataset = TEMPLATE[template_key]
            organ_index = [organ for organ in organ_target if organ not in organ_dataset]
            case_name = all_lines[line].strip().split()[0].split('.')[0].split('/')[-1]
            name_id.append(case_name)
            ct_path = os.path.join(args.data_root_path,dataset_name,case_name,'ct.nii.gz')
            case_path = os.path.join(args.data_root_path,dataset_name,case_name)
            avg_path = os.path.join(case_path,'average')
            avg_seg_path = os.path.join(avg_path,'segmentations')
            if not os.path.isdir(avg_seg_path):
                os.makedirs(avg_seg_path) 
            if len(args.model_list) == 1:
                file_copy_from_path = os.path.join(case_path,'backbones',args.model_list[0])
                pseudo_label_copy_from = os.path.join(file_copy_from_path,'pseudo_label.nii.gz')
                shutil.copy(pseudo_label_copy_from,os.path.join(avg_path,'pseudo_label.nii.gz'))
                organ_copy_from_list = os.listdir(os.path.join(file_copy_from_path,'segmentations'))
                for organ_copy in organ_copy_from_list:
                    organ_copy_from_path = os.path.join(file_copy_from_path,'segmentations',organ_copy)
                    shutil.copy(organ_copy_from_path,os.path.join(avg_seg_path,organ_copy))
                print('finish file copy for %s'%(case_name))
            
            ct_load = nib.load(ct_path)
            ct_data = ct_load.get_fdata()
            W,H,D = ct_data.shape
            affine_temp = ct_load.affine
            print('start create attention for %s'%(case_name))

            attention_overall = np.zeros((W,H,D))

            for idx in tqdm(range(len(organ_index))):
                organ_name = ORGAN_NAME_LOW[organ_index[idx]-1]
                consistency_map = np.zeros((len(args.model_list),W,H,D))
                if len(args.model_list) == 1:
                    consistency_map = np.zeros((W,H,D))
                entropy_map = np.zeros((W,H,D))
                overlap_initial = np.zeros((W,H,D))
                aveg_seg_data = nib.load(os.path.join(avg_path,'segmentations',organ_name+'.nii.gz')).get_fdata()


                for model_idx in range(len(args.model_list)):
                    organ_soft_pred_path = os.path.join(case_path,'backbones',args.model_list[model_idx],'soft_pred')
                    organ_entropy_path = os.path.join(case_path,'backbones',args.model_list[model_idx],'entropy')
                    if len(args.model_list) != 1:
                        organ_soft_pred = nib.load(os.path.join(organ_soft_pred_path,organ_name+'.nii.gz')).get_fdata()
                        consistency_map[model_idx] = organ_soft_pred/255
                    organ_entropy = nib.load(os.path.join(organ_entropy_path,organ_name+'.nii.gz')).get_fdata()
                    entropy_map += organ_entropy/255
                if len(args.model_list) != 1:
                    std_raw = np.std(consistency_map,axis=0)
                    std_float,std_binary = std_post_process(std_raw)
                else: 
                    std_float = consistency_map
                    std_binary = consistency_map
                entropy_raw = entropy_map/len(args.model_list)
                entropy_float,entropy_binary = entropy_post_process(entropy_raw)

                if args.save_consistency and len(args.model_list) != 1:
                    consistency_save_path = os.path.join(avg_path,'inconsistency')
                    if not os.path.isdir(consistency_save_path):
                        os.makedirs(consistency_save_path)
                    std_save = nib.Nifti1Image((std_float*255).astype(np.uint8),affine_temp)
                    nib.save(std_save,os.path.join(consistency_save_path,organ_name+'.nii.gz'))
                    print('%s inconsistency saved'%(organ_name))

                if args.save_entropy:
                    entropy_save_path = os.path.join(avg_path,'uncertainty')
                    if not os.path.isdir(entropy_save_path):
                        os.makedirs(entropy_save_path)
                    entropy_save = nib.Nifti1Image((entropy_float*255).astype(np.uint8),affine_temp)
                    nib.save(entropy_save,os.path.join(entropy_save_path,organ_name+'.nii.gz'))
                    print('%s uncertainty saved'%(organ_name))

                for surrounding_organ in ORGAN_NAME_OVERLAP:
                    if surrounding_organ != organ_name:
                        surrounding_organ_path = os.path.join(avg_path,'segmentations',surrounding_organ+'.nii.gz')
                        surrounding_organ_data = nib.load(surrounding_organ_path).get_fdata()
                        target_surrounding_sum = aveg_seg_data+surrounding_organ_data
                        overlap = target_surrounding_sum >1
                        overlap = overlap.astype(np.uint8)
                        overlap_initial += overlap
                
                overlap_total = overlap_initial > 0
                overlap_total = overlap_total.astype(np.uint8)
                
                if args.save_overlap:
                    overlap_save_path = os.path.join(avg_path,'overlap')
                    if not os.path.isdir(overlap_save_path):
                        os.makedirs(overlap_save_path)
                    overlap_save = nib.Nifti1Image(overlap_total,affine_temp)
                    nib.save(overlap_save,os.path.join(overlap_save_path,organ_name+'.nii.gz'))
                    print('%s overlap saved'%(organ_name))

                attention = std_binary + entropy_binary + overlap_total
                attention_binary = attention > 0
                attention_binary = attention_binary.astype(np.uint8)
                attention_heatmap = std_float + entropy_float + overlap_total
                attention_heatmap = attention_heatmap/np.max(attention_heatmap)

                attention_save_path = os.path.join(avg_path,'attention')
                if not os.path.isdir(attention_save_path):
                    os.makedirs(attention_save_path)
                attention_save = nib.Nifti1Image((attention_heatmap*255).astype(np.uint8),affine_temp)
                nib.save(attention_save,os.path.join(attention_save_path,organ_name+'.nii.gz'))
                print('%s attention saved'%(organ_name))
                
                attention_overall += attention_binary
            
            attention_value.append(np.sum(attention_overall))
    
    attention_value = np.array(attention_value)
    attention_value_sort = np.argsort(-attention_value)
    for i in attention_value_sort:
        sorted_name_id.append(name_id[i])
    print('case sorted complete')
    return sorted_name_id

def priority_list(sorted_name_id,args):
    for item in args.dataset_list:
        with open(os.path.join(args.data_txt_path,item + '.txt'), "r") as f:
            all_lines = f.readlines()
        dataset_name = all_lines[0].strip().split()[0].split('/')[0]
        csv_save_path = os.path.join(args.data_root_path,dataset_name)
        if os.path.isfile(os.path.join(csv_save_path,args.priority_name+'.csv')):
            os.remove(os.path.join(csv_save_path,args.priority_name+'.csv'))
        for case_name in sorted_name_id:
            row = [case_name]
            organ_attention = []
            organ = []
            non_zero_attention = []
            non_zero_organ = []
            sorted_organ = []
            attention_path = os.path.join(csv_save_path,case_name,'average','attention')
            organ_target = TEMPLATE['target']
            if int(dataset_name[0:2]) == 10:
                template_key = get_key(all_lines[0].strip().split()[0].split('.')[0])
            else: 
                template_key = get_key(dataset_name)
            organ_dataset = TEMPLATE[template_key]
            organ_index = [i for i in organ_target if i not in organ_dataset]

            for idx in organ_index:
                organ_name = ORGAN_NAME_LOW[idx-1]
                organ_attention_data = nib.load(os.path.join(attention_path,organ_name+'.nii.gz')).get_fdata()
                organ_attention_value = np.sum(organ_attention_data)
                organ_attention.append(organ_attention_value)
                organ.append(organ_name)
            
            for att,org in zip(organ_attention,organ):
                if att != 0:
                    non_zero_attention.append(att)
                    non_zero_organ.append(org)

            non_zero_attention = np.array(non_zero_attention)
            non_zero_attention_sorted = np.argsort(-non_zero_attention)
            
            for attention_idx in non_zero_attention_sorted:
                sorted_organ.append(non_zero_organ[attention_idx])

            for organ_csv in sorted_organ:
                row.append(organ_csv)
            print(row)

            with open(os.path.join(csv_save_path,args.priority_name+'.csv'),'a',newline='') as f:
                writer = csv.writer(f,delimiter=',', quotechar='"')
                writer.writerow(row)

























def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_path', default='/ccvl/net/ccvl15/chongyu/LargePseudoDataset/', help='atlas 8K data root path')
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT_123457891213', 'PAOT_10_inner'])
    parser.add_argument('--model_list',nargs='+', default=['swinunetr', 'unet','nnunet'])
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')
    parser.add_argument('--save_consistency', action="store_true", default=False, help='whether save consistency')
    parser.add_argument('--save_entropy', action="store_true", default=False, help='whether save binary entropy')
    parser.add_argument('--save_overlap', action="store_true", default=False, help='whether save overlap')
    parser.add_argument('--priority', action="store_true", default=False, help='whether save priority list')
    parser.add_argument('--priority_name', default='priority', help='priority csv name')
    args = parser.parse_args()

    sorted_name_id = create_attention(args)
    if args.priority:
        priority_list(sorted_name_id,args)

if __name__ == "__main__":
    main()