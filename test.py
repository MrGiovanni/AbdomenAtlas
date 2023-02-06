import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time
import shutil
import nibabel as nib
import csv

from monai.losses import DiceCELoss
from monai.data import load_decathlon_datalist, decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from model.SwinUNETR_partial import SwinUNETR
from dataset.dataloader_test import get_loader
from utils import loss
from utils.utils import dice_score, threshold_organ, visualize_label, merge_label, get_key, psedu_label_all_organ, psedu_label_single_organ, save_organ_label
from utils.utils import TEMPLATE, ORGAN_NAME, NUM_CLASS,ORGAN_NAME_LOW
from utils.utils import organ_post_process, threshold_organ

torch.multiprocessing.set_sharing_strategy('file_system')


def validation(model, ValLoader, val_transforms, args):
    save_dir = args.log_name
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model.eval()
    dice_list = {}
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count
    for index, batch in enumerate(tqdm(ValLoader)):
        image, label, name,name_img = batch["image"].cuda(), batch["post_label"], batch["name"],batch["name_img"]
        image_file_path = args.data_root_path + name_img[0] +'.nii.gz'
        print(image_file_path)
        print(image.shape)
        print(name)
        print(name_img)
        with torch.no_grad():
            pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.75, mode='gaussian')
            pred_sigmoid = F.sigmoid(pred)
        
        pred_hard = threshold_organ(pred_sigmoid)
        pred_hard = pred_hard.cpu()
        torch.cuda.empty_cache()

        B = pred_hard.shape[0]
        for b in range(B):
            content = 'case%s| '%(name[b])
            template_key = get_key(name[b])
            organ_list = TEMPLATE[template_key]
            organ_list_all = TEMPLATE['all'] # post processing all organ
            pred_hard_post,total_anomly_slice_number = organ_post_process(pred_hard.numpy(), organ_list_all,save_dir + '/' + name[0].split('/')[0]+'/'+ name_img[0].split('/')[-1])
            pred_hard_post = torch.tensor(pred_hard_post)
        
        if args.store_result:
            organ_index_all = TEMPLATE['all']
            for organ_index in organ_index_all:
                psedu_label_single = psedu_label_single_organ(pred_hard_post,organ_index)
                organ_name = ORGAN_NAME_LOW[organ_index-1]
                batch[organ_name]=psedu_label_single.cpu()
                
                save_organ_label(batch, save_dir + '/' + name[0].split('/')[0]+'/'+ name_img[0].split('/')[-1]+'/segmentations' , val_transforms,organ_index)

                old_name = os.path.join(save_dir, name[0].split('/')[0], name_img[0].split('/')[-1], 'segmentations', name_img[0].split('/')[-1]+'_'+organ_name+'.nii.gz')
                new_name = os.path.join(save_dir, name[0].split('/')[0], name_img[0].split('/')[-1], 'segmentations', organ_name+'.nii.gz')

                os.rename(old_name,new_name)

            psedu_label_all = psedu_label_all_organ(pred_hard_post)
            batch['psedu_label'] = psedu_label_all.cpu()
            visualize_label(batch, save_dir + '/' + name[0].split('/')[0] , val_transforms)

            old_name = os.path.join(save_dir + '/' + name[0].split('/')[0], name_img[0].split('/')[-1], name_img[0].split('/')[-1] + '_original_label.nii.gz')
            new_name = os.path.join(save_dir + '/' + name[0].split('/')[0], name_img[0].split('/')[-1], 'original_label.nii.gz')
            os.rename(old_name,new_name)

            old_name = os.path.join(save_dir + '/' + name[0].split('/')[0], name_img[0].split('/')[-1], name_img[0].split('/')[-1] + '_pseudo_label.nii.gz')
            new_name = os.path.join(save_dir + '/' + name[0].split('/')[0], name_img[0].split('/')[-1], 'pseudo_label.nii.gz')
            os.rename(old_name,new_name)

            destination = save_dir + '/' + name[0].split('/')[0]+'/'+ name_img[0].split('/')[-1]+'/ct.nii.gz'
            try:
                shutil.copy(image_file_path, destination)
                print("Image File copied successfully.")
            except:
                print("Error occurred while copying file.")
            right_lung_data_path = save_dir + '/' + name[0].split('/')[0]+'/'+ name_img[0].split('/')[-1]+'/segmentations/lung_right.nii.gz'
            left_lung_data_path = save_dir + '/' + name[0].split('/')[0]+'/'+ name_img[0].split('/')[-1]+'/segmentations/lung_left.nii.gz'
            organ_name=['lung_right','lung_left']
            ct_data = nib.load(destination).get_fdata()
            right_lung_data = nib.load(right_lung_data_path).get_fdata()
            left_lung_data = nib.load(left_lung_data_path).get_fdata()
            right_lung_data_sum = np.sum(right_lung_data,axis=(0,1))
            left_lung_data_sum = np.sum(left_lung_data,axis=(0,1))
            right_lung_size = np.sum(right_lung_data,axis=(0,1,2))
            left_lung_size = np.sum(left_lung_data,axis=(0,1,2))
            if right_lung_size>left_lung_size:
                non_zero_idx = np.nonzero(right_lung_data_sum)
                first_non_zero_idx = non_zero_idx[0][0]
                if total_anomly_slice_number!=0:
                    for s in range(first_non_zero_idx,right_lung_data.shape[-1]):
                        if len(np.unique(ct_data[:,:,s]))!= 1 and right_lung_data_sum[s] ==0:
                            print('start writing csv as slice: '+str(s+1))
                            with open(save_dir + '/' + name[0].split('/')[0]+'/anomaly.csv','a',newline='') as f:
                                writer = csv.writer(f)
                                content = name_img[0].split('/')[-1]
                                writer.writerow([content,organ_name[0],s+1])
                                writer.writerow([content,organ_name[1],s+1])
            else: 
                non_zero_idx = np.nonzero(left_lung_data_sum)
                first_non_zero_idx = non_zero_idx[0][0]
                if total_anomly_slice_number!=0:
                    for s in range(first_non_zero_idx,left_lung_data.shape[-1]):
                        if len(np.unique(ct_data[:,:,s]))!= 1 and left_lung_data_sum[s] ==0:
                            print('start writing csv as slice: '+str(s+1))
                            with open(save_dir + '/' + name[0].split('/')[0]+'/anomaly.csv','a',newline='') as f:
                                writer = csv.writer(f)
                                content = name_img[0].split('/')[-1]
                                writer.writerow([content,organ_name[0],s+1])
                                writer.writerow([content,organ_name[1],s+1])
                
        torch.cuda.empty_cache()
    
    ave_organ_dice = np.zeros((2, NUM_CLASS))

def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='Nvidia/old_fold0', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--resume', default='./out/Nvidia/old_fold0/aepoch_500.pth', help='The path resume from checkpoint')
    parser.add_argument('--pretrain', default='./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt', 
                        help='The path of pretrain model')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=1000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')

    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT_123457891213', 'PAOT_10_inner']) # 'PAOT', 'felix'
    ### please check this argment carefully
    ### PAOT: include PAOT_123457891213 and PAOT_10
    ### PAOT_123457891213: include 1 2 3 4 5 7 8 9 12 13
    ### PAOT_10_inner: same with NVIDIA for comparison
    ### PAOT_10: original division
    parser.add_argument('--data_root_path', default='/home/jliu288/data/whole_organ/', help='data root path')
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')

    parser.add_argument('--phase', default='test', help='train or validation or test')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--store_result', action="store_true", default=False, help='whether save prediction result')
    parser.add_argument('--cache_rate', default=0.6, type=float, help='The percentage of cached data in total')

    parser.add_argument('--threshold_organ', default='Pancreas Tumor')
    parser.add_argument('--threshold', default=0.6, type=float)

    args = parser.parse_args()

    # prepare the 3D model
    model = SwinUNETR(img_size=(args.roi_x, args.roi_y, args.roi_z),
                      in_channels=1,
                      out_channels=NUM_CLASS,
                      feature_size=48,
                      drop_rate=0.0,
                      attn_drop_rate=0.0,
                      dropout_path_rate=0.0,
                      use_checkpoint=False,
                      encoding='word_embedding'
                     )
    
    #Load pre-trained weights
    store_dict = model.state_dict()
    checkpoint = torch.load(args.resume)
    load_dict = checkpoint['net']
    # args.epoch = checkpoint['epoch']

    for key, value in load_dict.items():
        name = '.'.join(key.split('.')[1:])
        store_dict[name] = value

    model.load_state_dict(store_dict)
    print('Use pretrained weights')

    model.cuda()

    torch.backends.cudnn.benchmark = True

    test_loader, val_transforms = get_loader(args)

    validation(model, test_loader, val_transforms, args)

if __name__ == "__main__":
    main()
