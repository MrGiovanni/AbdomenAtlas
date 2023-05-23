# Calculate the dice using orignal label and pseudo label
import numpy as np
import nibabel as nib
import os
from tqdm import tqdm
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/medical_backup/LargePseudoDataset', help='The path of your data')
    parser.add_argument('--dataset_name', default='18_FLARE23', help='The dataset name for generating')
    parser.add_argument('--organs', nargs='+', default=['liver','kidney_right','spleen','pancreas','aorta','postcava',
                       'adrenal_gland_right','adrenal_gland_left','gall_bladder','esophagus','stomach','duodenum','kidney_left'], help='folder to filter the files(img,train,imagesTr)')
    parser.add_argument('--backbone', default='swinunetr', help='backbone [swinunetr or unet]')
    parser.add_argument('--out', default='./Dice')
    parser.add_argument('--save_file', default='dice.txt')
    parser.add_argument('--ground_truth', default = {'aorta':1,'adrenal_gland_right':2,'adrenal_gland_left':2,'colon':6,'duodenum':7,'gall_bladder':8,'postcava':9,
                    'kidney_left':10,'kidney_right':11,'liver':12,'pancreas':13,'spleen':18,'stomach':19})

    args = parser.parse_args()
    organs_set = args.ground_truth
    dice_dic = {}
    id = []
    for organ in args.organs:
        if organ not in dice_dic:
            dice_dic[organ] = []
    file_list = os.listdir(os.path.join(args.data_path,args.dataset_name))
    for files in file_list:
        if files.startswith('.') or files.endswith('.csv'):
            file_list.remove(files)
    file_list = sorted(file_list)
    Num = len(file_list)
    for i in tqdm(range(Num)):
        true_file = os.path.join(args.data_path,args.dataset_name,file_list[i],'original_label.nii.gz')
        if os.path.exists(true_file):
            img_true = nib.load(true_file)
            data_true = img_true.get_fdata()
        else:
            continue
        for k in dice_dic.keys():
            label = organs_set[k]
            pred_file = os.path.join(args.data_path,args.dataset_name,file_list[i],'backbones',args.backbone,'segmentations',k+'.nii.gz')
            img_pred = nib.load(pred_file)
            data_pred = img_pred.get_fdata()
            y_true_class = (data_true == label).astype(bool)
            if y_true_class.sum()==0:
                continue
            y_pred_class = (data_pred == 1).astype(bool)
            intersection = np.logical_and(y_true_class, y_pred_class).sum()
            smooth = 1e-4
            dice = (2. * intersection+smooth) / (y_true_class.sum() + y_pred_class.sum()+smooth)
            dice_dic[k].append(dice)
    re_dic = {}
    for k in dice_dic.keys():
        re_dic[k] = np.sum(dice_dic[k])/len(dice_dic[k])

    with open(os.path.join(args.out,args.save_file),'w') as file:
        for key, value in re_dic.items():
            file.write(f'{key}: {value}\n')
        file.close()
    Dice_df = pd.DataFrame(dice_dic)
    Dice_df.insert(0,'Case_Id',id)
    Dice_df.to_csv(os.path.join(args.out,'dice.csv'), index=False)
    

if __name__ == "__main__":
    main()

