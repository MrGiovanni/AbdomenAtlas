import numpy as np
import nibabel as nib
import os
from tqdm import tqdm
import argparse
import shutil
import cc3d

TEMPLATE = {'spleen':1,'kidney_right':2,'kidney_left':3,'gall_bladder':4,'esophagus':5,'liver':6,'stomach':7,
                    'aorta':8,'postcava':9,'portal_vein_and_splenic_vein':10,'pancreas':11,'adrenal_gland_right':12,'adrenal_gland_left':13,
                    'duodenum':14,'hepatic_vessel':15,'lung_right':16,'lung_left':17,'colon':18,'intestine':19,'rectum':20,'bladder':21,'prostate':22,
                    'femur_left':23,'femur_right':24,'celiac_truck':25,'kidney_tumor':26,'liver_tumor':27,'pancreas_tumor':28,'hepatic_vessel_tumor':29,
                    'lung_tumor':30,'colon_tumor':31,'kidney_cyst':32}
TEMPLATE_orgianl={
    '01': [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
    '02': [1,0,3,4,5,6,7,0,0,0,11,0,0,14],
    '03': [6],
    '04': [6,27], 
    '05': [2,26,32],
    '07': [6,1,3,2,7,4,5,11,14,18,19,12,20,21,23,24],
    '08': [6, 2, 1, 11],
    '09': [1,2,3,4,5,6,7,8,9,11,12,13,14,21,22],
    '12': [6,21,16,2],  
    '13': [6,2,1,11,8,9,7,4,5,12,13,25], 
    '14': [8,12,0,0,0,18,14,4,9,3,2,6,11,28,0,0,0,1,7,10],
    '18': [6,2,1,11,8,9,12,13,4,5,7,14,3],
    '10_03': [6, 27],
    '10_06': [30],
    '10_07': [11, 28],
    '10_08': [15, 29],
    '10_09': [1],
    '10_10': [31],
    'all': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
}

def rl_split(original_data, organ_index, right_index, left_index):
    RIGHT_ORGAN = right_index
    LEFT_ORGAN = left_index
    label_raw = original_data.copy()
    label_in = np.zeros(label_raw.shape)
    label_in[label_raw == organ_index] = 1
    label_out = cc3d.connected_components(label_in, connectivity=26)

    if len(np.unique(label_out)) > 3:
        count_sum = 0
        values, counts = np.unique(label_out, return_counts=True)
        num_list_sorted = sorted(values, key=lambda x: counts[x])[::-1]
        for i in num_list_sorted[3:]:
            label_out[label_out==i] = 0
            count_sum += counts[i]
        label_new = np.zeros(label_out.shape)
        for tgt, src in enumerate(num_list_sorted[:3]):
            label_new[label_out==src] = tgt
        label_out = label_new

    a1,b1,c1 = np.where(label_out==1)
    a2,b2,c2 = np.where(label_out==2)
    
    label_new = np.zeros(label_out.shape)
    original_data[original_data == organ_index]=0

    if np.mean(a1) < np.mean(a2):
        label_raw [label_out==1] = LEFT_ORGAN
        label_raw [label_out==2] = RIGHT_ORGAN
    else:
        label_raw [label_out==1] = RIGHT_ORGAN
        label_raw [label_out==2] = LEFT_ORGAN

    return label_raw 
def Split(args,original_data):
    dataset_index = args.dataset_name[:2]
    if dataset_index == '02':
        print("02")
        organ_index = 2
        right_index =2 
        left_index =3
        original_data = rl_split(original_data, organ_index, right_index, left_index)

    if dataset_index == '05':
        print("05")
        organ_index = 2
        right_index =2 
        left_index =3
        original_data = rl_split(original_data, organ_index, right_index, left_index)
    
    if dataset_index == '08':
        print("08")
        organ_index = 2
        right_index =2 
        left_index =3
        original_data = rl_split(original_data, organ_index, right_index, left_index)
    
    if dataset_index == '13':
        print("13")
        organ_index = 2
        right_index =2 
        left_index =3
        original_data = rl_split(original_data, organ_index, right_index, left_index)
   
    if dataset_index == '12':
        print('12')
        organ_index = 16
        right_index = 16 
        left_index = 17
        input_data = rl_split(original_data, organ_index, right_index, left_index)
        print('Lung_Split')
        organ_index = 2
        right_index =2 
        left_index =3
        original_data = rl_split(input_data, organ_index, right_index, left_index)
    if dataset_index == '07':
        print("07")
        organ_index = 12
        right_index = 12 
        left_index = 13
        original_data = rl_split(original_data, organ_index, right_index, left_index)
    if dataset_index == '14':
        organ_index = 12
        right_index = 12 
        left_index = 13
        original_data = rl_split(original_data, organ_index, right_index, left_index)
    return original_data


def label_transfer(args,original_data,case):
    
    if args.dataset_name[:2] != '10':
        original_index = TEMPLATE_orgianl[args.dataset_name[:2]]
    elif args.dataset_name[:2] == '10':
        if case.split('_')[0] == 'colon':
            original_index = TEMPLATE_orgianl['10_10']
        elif case.split('_')[0] == 'hepaticvessel':
            original_index = TEMPLATE_orgianl['10_08']
        elif case.split('_')[0] == 'liver':
            original_index = TEMPLATE_orgianl['10_03']
        elif case.split('_')[0] == 'lung':
            original_index = TEMPLATE_orgianl['10_06']
        elif case.split('_')[0] == 'pancreas':
            original_index = TEMPLATE_orgianl['10_07']
        elif case.split('_')[0] == 'spleen':
            original_index = TEMPLATE_orgianl['10_09']
    
    data_index = np.unique(original_data)
    our_original_data = np.zeros(original_data.shape)
    for i in range(len(original_index)):
        index = original_index[i]
        temp = np.where(original_data == i+1)
        our_original_data[temp] = index
    return our_original_data


def generate_label(args,original_label_file,pseudo_label_file, case,destination_path):
    temp_path = os.path.join(args.data_path,args.dataset_name,args.subfolder,case)
    pseudo_label = nib.load(pseudo_label_file)
    pseudo_label_data = pseudo_label.get_fdata()
    affine = pseudo_label.affine
    our_label = pseudo_label_data
    temp_files_list = os.listdir(os.path.join(temp_path,'average','segmentations'))
    revised_organ_list = [f for f in temp_files_list if f.endswith('_revised.nii.gz') and not f.startswith('.')]
    if os.path.exists(original_label_file):
        original_label = nib.load(original_label_file)
        original_label_data = original_label.get_fdata()
        # perform label_transfer
        our_original_data = label_transfer(args,original_label_data,case)

        original_label_data = Split(args,our_original_data)

        label_num_list = np.unique(original_label_data)
        label_num_list = label_num_list[1:]
    else:
        label_num_list = None

    if revised_organ_list is not None:
        for revised_organs in revised_organ_list:
            organ = revised_organs[:-15]
            revised_organs = nib.load(os.path.join(temp_path,'average','segmentations',revised_organs))
            revised_organs_data = revised_organs.get_fdata()
            our_label[pseudo_label_data==TEMPLATE[organ]]=0
            our_label[revised_organs_data==1]=TEMPLATE[organ]
    if label_num_list is not None:
        for labels in label_num_list:
            our_label[our_label==labels]=0
            our_label[original_label_data==labels]=labels


    
    our_label = nib.Nifti1Image(our_label, affine=affine)
    filename = os.path.join(destination_path,'pseudo_label.nii.gz')
    nib.save(our_label, filename)



def main_process(args,cases_list):
    for i in tqdm(range(len(cases_list))):
        print(cases_list[i])
        # Generate the folder for our dataset.
        if args.subfolder == '':
            destination_path = os.path.join(args.save_dir,args.dataset_name+'_'+cases_list[i])
            seg_path = os.path.join(args.save_dir,args.dataset_name+'_'+cases_list[i],'segmentations')
        else:
            destination_path = os.path.join(args.save_dir,args.dataset_name+'_'+args.subfolder+'_'+cases_list[i])
            seg_path = os.path.join(args.save_dir,args.dataset_name+'_'+args.subfolder+'_'+cases_list[i],'segmentations')
        if not os.path.exists(destination_path):
            os.mkdir(destination_path)
        # Copy the ct from the revised data
        ct_source_file = os.path.join(args.data_path,args.dataset_name,args.subfolder,cases_list[i],'ct.nii.gz')
        ct_destination_file = os.path.join(destination_path,'ct.nii.gz')
        if not os.path.exists(ct_destination_file):
            shutil.copy2(ct_source_file, ct_destination_file)

    
        or_source_file = os.path.join(args.data_path,args.dataset_name,args.subfolder,cases_list[i],'original_label.nii.gz')
        or_destination_file = os.path.join(destination_path,'original_label.nii.gz')
        if os.path.exists(or_source_file):
            shutil.copy2(or_source_file, or_destination_file)
        ps_source_file = os.path.join(args.data_path,args.dataset_name,cases_list[i],'backbones',args.backbone,'pseudo_label.nii.gz')
        ps_destination_file = os.path.join(destination_path,'pseudo_label_'+args.version+'.nii.gz')
        if os.path.exists(ps_source_file):
            shutil.copy2(ps_source_file, ps_destination_file)

        generate_label(args,or_source_file,ps_source_file, cases_list[i], destination_path)

        if not os.path.exists(seg_path):
            os.mkdir(seg_path)
        temp_path = os.path.join(args.data_path,args.dataset_name,cases_list[i],'average','segmentations')
        temp_files_list = os.listdir(temp_path)
        revised_organ_list = [f[:-15] for f in temp_files_list if f.endswith('_revised.nii.gz') and not f.startswith('.')]
        for f in temp_files_list:
            organ = f[:-7]
            if organ in revised_organ_list:
                shutil.copy2(os.path.join(temp_path,organ+'.nii.gz'), os.path.join(seg_path,organ+'_'+args.version+'.nii.gz'))
                shutil.copy2(os.path.join(temp_path,organ+'_revised.nii.gz'), os.path.join(seg_path,organ+'.nii.gz'))
            else:
                if not organ.endswith('_revised'):
                    shutil.copy2(os.path.join(temp_path,organ+'.nii.gz'), os.path.join(seg_path,organ+'.nii.gz'))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/ccvl/net/ccvl15/zzhou82/LargePseudoDataset', help='The path of your data')
    parser.add_argument('--dataset_name', default='14_FELIX', help='The dataset name')
    parser.add_argument('--subfolder', default='', help='The subfolder name')
    parser.add_argument('--backbone', default='swinunetr', help='The backbone')
    parser.add_argument('--save_dir', default='/ccvl/net/ccvl15/tzhang85/AbdomenAtlas_8K_internal', help='The saving path')
    parser.add_argument('--version', default='V1', help='The version of revised label')

    args = parser.parse_args()

    save_dir = os.path.join(args.save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    else:
        print('The saving directory exists')


    cases_list = os.listdir(os.path.join(args.data_path,args.dataset_name,args.subfolder))
    cases_list = [f for f in cases_list if not f.startswith('.') and not f.endswith('.csv')]
    main_process(args,cases_list)

if __name__ == "__main__":
    main()
