import numpy as np
import nibabel as nib
import os
from tqdm import tqdm
import argparse
import pandas as pd
import shutil

TEMPLATE = {'spleen':1,'kidney_right':2,'kidney_left':3,'gall_bladder':4,'esophagus':5,'liver':6,'stomach':7,
                    'aorta':8,'postcava':9,'portal_vein_and_splenic_vein':10,'pancreas':11,'adrenal_gland_right':12,'adrenal_gland_left':13,
                    'duodenum':14,'hepatic_vessel':15,'lung_right':16,'lung_left':17,'colon':18,'intestine':19,'rectum':20,'bladder':21,'prostate':22,
                    'femur_left':23,'femur_right':24,'celiac_truck':25,'kidney_tumor':26,'liver_tumor':27,'pancreas_tumor':28,'hepatic_vessel_tumor':29,
                    'lung_tumor':30,'colon_tumor':31,'kidney_cyst':32}

def generate_label(args,original_label_file,pseudo_label_file, case,destination_path):
    temp_path = os.path.join(args.data_path,args.dataset_name,case)
    pseudo_label = nib.load(pseudo_label_file)
    pseudo_label_data = pseudo_label.get_fdata()
    affine = pseudo_label.affine
    our_label = pseudo_label_data
    temp_files_list = os.listdir(os.path.join(temp_path,'average','segmentations'))
    revised_organ_list = [f for f in temp_files_list if f.endswith('_revised.nii.gz') and not f.startswith('.')]
    if os.path.exists(original_label_file):
        original_label = nib.load(original_label_file)
        original_label_data = original_label.get_fdata()
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
        destination_path = os.path.join(args.save_dir,args.dataset_name+'_'+args.subfolder+'_'+cases_list[i])
        if not os.path.exists(destination_path):
            os.mkdir(destination_path)
        ct_source_file = os.path.join(args.data_path,args.dataset_name,cases_list[i],'ct.nii.gz')
        ct_destination_file = os.path.join(destination_path,'ct.nii.gz')
        if args.original_label:
            or_source_file = os.path.join(args.data_path,args.dataset_name,cases_list[i],'original_label.nii.gz')
            or_destination_file = os.path.join(destination_path,'original_label.nii.gz')
        if args.pseudo_label:
            ps_source_file = os.path.join(args.data_path,args.dataset_name,cases_list[i],'backbones',args.backbone,'pseudo_label.nii.gz')
            ps_destination_file = os.path.join(destination_path,'pseudo_label_'+args.version+'.nii.gz')
        generate_label(args,or_source_file,ps_source_file, cases_list[i], destination_path)
        if not os.path.exists(ct_destination_file):
            shutil.copy2(ct_source_file, ct_destination_file)
        if os.path.exists(or_source_file):
            shutil.copy2(or_source_file, or_destination_file)
        if os.path.exists(ps_source_file):
            shutil.copy2(ps_source_file, ps_destination_file)
        destination_path = os.path.join(args.save_dir,args.dataset_name+'_'+args.subfolder+'_'+cases_list[i],'segmentations')
        if not os.path.exists(destination_path):
            os.mkdir(destination_path)
        temp_path = os.path.join(args.data_path,args.dataset_name,cases_list[i],'average','segmentations')
        temp_files_list = os.listdir(temp_path)
        revised_organ_list = [f[:-15] for f in temp_files_list if f.endswith('_revised.nii.gz') and not f.startswith('.')]
        for f in temp_files_list:
            organ = f[:-7]
            if organ in revised_organ_list:
                shutil.copy2(os.path.join(temp_path,organ+'.nii.gz'), os.path.join(destination_path,organ+'_'+args.version+'.nii.gz'))
                shutil.copy2(os.path.join(temp_path,organ+'_revised.nii.gz'), os.path.join(destination_path,organ+'.nii.gz'))
            else:
                if not organ.endswith('_revised'):
                    shutil.copy2(os.path.join(temp_path,organ+'.nii.gz'), os.path.join(destination_path,organ+'.nii.gz'))
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/data2/tzhang/M_data', help='The path of your data')
    parser.add_argument('--dataset_name', default='02_TCIA_Pancreas-CT', help='The dataset name')
    parser.add_argument('--subfolder', default='img', help='The subfolder name')
    parser.add_argument('--backbone', default='swinunetr', help='The backbone')
    parser.add_argument('--save_dir', default='/mnt/tiezheng/AbdomenAtlas_8K_internal', help='The saving path')
    parser.add_argument('--pseudo_label', default=True, help='Whether have pseudo label')
    parser.add_argument('--original_label', default=True, help='Whether have orginal label')
    parser.add_argument('--version', default='V1', help='The version of revised label')

    args = parser.parse_args()

    save_dir = os.path.join(args.save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    else:
        print('The saving directory exists')


    cases_list = os.listdir(os.path.join(args.data_path,args.dataset_name))
    cases_list = [f for f in cases_list if not f.startswith('.')]
    main_process(args,cases_list)

if __name__ == "__main__":
    main()

