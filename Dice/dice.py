# Calculate the dice using orignal label and pseudolabel
import numpy as np
import nibabel as nib
import os
from tqdm import tqdm



def dice_coefficient(y_true,y_pred,label):
    y_true_class = (y_true == label).astype(bool)
    y_pred_class = (y_pred == 1).astype(bool)

    intersection = np.logical_and(y_true_class, y_pred_class).sum()
    smooth = 1e-4
    dice = (2. * intersection+smooth) / (y_true_class.sum() + y_pred_class.sum()+smooth)
    
    return dice

def main():
    path = '/data/zzhou82/dataset/LargePseudoDataset/14_FELIX'
    file_list = os.listdir(path)
    Num = len(file_list)
    organs = ["aorta","colon","duodenum","gall_bladder","kidney_left","kidney_right","liver","pancreas","spleen","stomach",
            "portal_vein_and_splenic_vein"]
    num_class = [1,6,7,8,10,11,12,13,18,19,20]

    dice_scores = np.zeros((1,len(num_class)))
    for i in tqdm(range(Num)):
        dice_score = []
        true_file = os.path.join(path+'/'+file_list[i]+'/original_label.nii.gz')
        img_true = nib.load(true_file)
        data_true = img_true.get_fdata()
        for j in range(len(num_class)):
            label  = num_class[j]
            pred_file = os.path.join(path+'/'+file_list[i]+'/segmentations/'+organs[j]+'.nii.gz')
            img_pred = nib.load(pred_file)
            data_pred = img_pred.get_fdata()
            dice = dice_coefficient(data_true,data_pred,label)
            dice_score.append(dice)
        dice_scores += dice_score
    dice_scores = dice_scores/Num
    print(dice_scores)
    

if __name__ == "__main__":
    main()
