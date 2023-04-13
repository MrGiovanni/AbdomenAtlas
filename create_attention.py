import os
import argparse
import numpy as np
import nibabel as nib
from utils.utils import ORGAN_NAME_OVERLAP,TEMPLATE,ORGAN_NAME_LOW,ORGAN_NAME_OVERLAP
from utils.utils import entropy_post_process,std_post_process
from tqdm import tqdm
import csv

def create_attention(args):
    organ_index = TEMPLATE['target']
    name_id = []
    attention_value = []
    sorted_name_id = []
    for item in args.dataset_list:
        for line in open(os.path.join(args.data_txt_path,item + '_test.txt')):
            dataset_name = line.strip().split()[0].split('/')[0]
            case_name = line.strip().split()[0].split('.')[0].split('/')[-1]
            name_id.append(case_name)
            ct_path = os.path.join(args.data_root_path,dataset_name,case_name,'ct.nii.gz')
            case_path = os.path.join(args.data_root_path,dataset_name,case_name)
            avg_path = os.path.join(case_path,'average')


            ct_load = nib.load(ct_path)
            ct_data = ct_load.get_fdata()
            W,H,D = ct_data.shape
            affine_temp = ct_load.affine
            print('start create attention for %s'%(case_name))

            attention_overall = np.zeros((W,H,D))

            for idx in organ_index:
                organ_name = ORGAN_NAME_LOW[idx-1]
                consistency_map = np.zeros((len(args.model_list),W,H,D))
                entropy_map = np.zeros((W,H,D))
                overlap_initial = np.zeros((W,H,D))
                aveg_seg_data = nib.load(os.path.join(avg_path,'segmentations',organ_name+'.nii.gz')).get_fdata()


                for model_idx in range(len(args.model_list)):
                    organ_soft_pred_path = os.path.join(case_path,'backbones',args.model_list[model_idx],'soft_pred')
                    organ_entropy_path = os.path.join(case_path,'backbones',args.model_list[model_idx],'entropy')
                    organ_soft_pred = nib.load(os.path.join(organ_soft_pred_path,organ_name+'.nii.gz')).get_fdata()
                    organ_entropy = nib.load(os.path.join(organ_entropy_path,organ_name+'.nii.gz')).get_fdata()
                    consistency_map[model_idx] = organ_soft_pred/255
                    entropy_map += organ_entropy/255
                
                std_raw = np.std(consistency_map,axis=0)
                std_float,std_binary = std_post_process(std_raw)
                entropy_raw = entropy_map/len(args.model_list)
                entropy_float,entropy_binary = entropy_post_process(entropy_raw)

                if args.save_consistency:
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
    return dataset_name,sorted_name_id

def priority_list(dataset_name,sorted_name_id,args):
    csv_save_path = os.path.join(args.data_root_path,dataset_name)
    for case_name in sorted_name_id:
        row = [case_name]
        organ_attention = []
        organ = []
        non_zero_attention = []
        non_zero_organ = []
        sorted_organ = []
        attention_path = os.path.join(csv_save_path,case_name,'average','attention')
        organ_index = TEMPLATE['target']
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

        with open(os.path.join(csv_save_path,'priority.csv'),'a',newline='') as f:
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
    args = parser.parse_args()

    dataset_name,sorted_name_id = create_attention(args)
    priority_list(dataset_name,sorted_name_id,args)

if __name__ == "__main__":
    main()