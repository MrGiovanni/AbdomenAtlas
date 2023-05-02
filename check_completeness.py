# python -W ignore check_completeness.py --datalist PAOT --data_root_path /medical_backup/LargePseudoDataset/ --data_txt_path dataset/dataset_list/ --out checklist

import os
import argparse

def check_segmentations(args,dataset,existing_patient):
    segmentations = os.path.join(args.data_root_path,dataset,existing_patient,'backbones',args.backbone,'segmentations')
    organ_list = os.listdir(segmentations)
    for i in range(len(organ_list)):
        organ_list[i] = organ_list[i][:-7]
    if sorted(args.organs) == sorted(organ_list):
        return True
    return False

def check_list(args):
    true_dataset = os.path.join(args.data_txt_path, args.datalist + ".txt")
    organs = args.organs
    existing_id = []
    missing_id = []
    true_id = []
    with open(true_dataset, "r") as file:
        for lines in file:
            line = lines.strip().split('\t')[0].split('/')
            dataset = args.data_root_path + line[0]
            true_id.append(line[-1][:-7])
            missing_id.append(line[-1][:-7])

    for existing_patient in os.listdir(dataset):
        if not existing_patient.startswith(".") and not existing_patient.endswith(".csv"):
            if existing_patient in true_id:
                if check_segmentations(args,dataset,existing_patient):
                    existing_id.append(existing_patient)
                    missing_id.remove(existing_patient)
    return existing_id, missing_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_path', default='/medical_backup/LargePseudoDataset/', help='data root path')
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')
    parser.add_argument('--datalist', default='PAOT_01')
    parser.add_argument('--backbone', default='swinunetr')
    parser.add_argument('--organs',nargs='+',default = ['adrenal_gland_left','adrenal_gland_right','aorta','bladder','celiac_truck',
    'colon','colon_tumor','duodenum','esophagus','femur_left','femur_right','gall_bladder','hepatic_vessel',
    'hepatic_vessel_tumor','intestine','kidney_cyst','kidney_left','kidney_right','kidney_tumor','liver','liver_tumor','lung_left',
    'lung_right','lung_tumor','pancreas','pancreas_tumor','portal_vein_and_splenic_vein','postcava','prostate','rectum','spleen',
    'stomach'])
    parser.add_argument('--out', default='./checklist/')

    args = parser.parse_args()
    
    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    existing_id, missing_id = check_list(args)

    if missing_id :
        print("Some cases are missing, please cheack the missing list")
    else:
        print("There is no missing cases")

    with open(os.path.join(args.out, args.datalist+'_existing_id.txt'), "w") as file:
        for item in existing_id:
            file.write("%s\n" % item)
    with open(os.path.join(args.out, args.datalist+'_missing_id.txt'), "w") as file:
        for item in missing_id:
            file.write("%s\n" % item)

if __name__ == "__main__":
    main()
