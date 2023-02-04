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
        # os.makedirs(save_dir+'/predict')
    model.eval()
    dice_list = {}
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count
    for index, batch in enumerate(tqdm(ValLoader)):
        # print('%d processd' % (index))
        image, label, name,name_img = batch["image"].cuda(), batch["post_label"], batch["name"],batch["name_img"]
        image_file_path = args.data_root_path + name_img[0] +'.nii.gz'
        print(image_file_path)
        print(image.shape)
        print(name)
        print(name_img)
        # print(label.shape)
        with torch.no_grad():
            # with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.75, mode='gaussian')
            pred_sigmoid = F.sigmoid(pred)
        
        #pred_hard = threshold_organ(pred_sigmoid, organ=args.threshold_organ, threshold=args.threshold)
        pred_hard = threshold_organ(pred_sigmoid)
        # B = pred_hard.shape[0]
        # D = pred_hard.shape[-1]
        # for b in range(B):
        #     for d in range(D):
        #         if len(torch.unique(image[b,:,:,:,d]))==1:
        #             pred_hard[b,:,:,:,d]=0
        # print(torch.unique(image[:,:,:,:,290]))
        # print(torch.unique(image[:,:,:,:,291]))
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
            # np.savez_compressed(save_dir + '/' + name[0].split('/')[0] +'/'+ name[0].split('/')[-1], pred=pred_hard)

            # for organ in organ_list:
            #     if torch.sum(label[b,organ-1,:,:,:].cuda()) != 0:
            #         dice_organ, recall, precision = dice_score(pred_hard_post[b,organ-1,:,:,:].cuda(), label[b,organ-1,:,:,:].cuda())
            #         dice_list[template_key][0][organ-1] += dice_organ.item()
            #         dice_list[template_key][1][organ-1] += 1
            #         content += '%s: %.4f, '%(ORGAN_NAME[organ-1], dice_organ.item())
            #         print('%s: dice %.4f, recall %.4f, precision %.4f.'%(ORGAN_NAME[organ-1], dice_organ.item(), recall.item(), precision.item()))
            # print(content)
        
        if args.store_result:
            # pred_sigmoid_store = (pred_sigmoid.cpu().numpy() * 255).astype(np.uint8)
            # label_store = (label.numpy()).astype(np.uint8)
            # np.savez_compressed(save_dir + '/predict/' + name[0].split('/')[0] + name[0].split('/')[-1], 
            #                 pred=pred_sigmoid_store, label=label_store)
            ### testing phase for this function

            # one_channel_label_v1, one_channel_label_v2 = merge_label(pred_hard_post, name)
            organ_index_all = TEMPLATE['all']
            for organ_index in organ_index_all:
                psedu_label_single = psedu_label_single_organ(pred_hard_post,organ_index)
                organ_name = ORGAN_NAME_LOW[organ_index-1]
                batch[organ_name]=psedu_label_single.cpu()
                
                save_organ_label(batch, save_dir + '/' + name[0].split('/')[0]+'/'+ name_img[0].split('/')[-1]+'/segmentations' , val_transforms,organ_index)

                old_name = os.path.join(save_dir, name[0].split('/')[0], name_img[0].split('/')[-1], 'segmentations', name_img[0].split('/')[-1]+'_'+organ_name+'.nii.gz')
                new_name = os.path.join(save_dir, name[0].split('/')[0], name_img[0].split('/')[-1], 'segmentations', organ_name+'.nii.gz')

                print('>> RENAME: {} -> {}'.format(old_name, new_name))
                os.rename(old_name,new_name)

            psedu_label_all = psedu_label_all_organ(pred_hard_post)
            batch['psedu_label'] = psedu_label_all.cpu()
            # batch['one_channel_label_v1'] = one_channel_label_v1.cpu()
            # batch['one_channel_label_v2'] = one_channel_label_v2.cpu()

            # _, split_label = merge_label(batch["post_label"], name)
            # batch['split_label'] = split_label.cpu()
            # print(batch['label'].shape, batch['one_channel_label'].shape)
            # print(torch.unique(batch['label']), torch.unique(batch['one_channel_label']))
            visualize_label(batch, save_dir + '/' + name[0].split('/')[0] , val_transforms)

            old_name = os.path.join(save_dir + '/' + name[0].split('/')[0], name_img[0].split('/')[-1] + '_original_label.nii.gz')
            new_name = os.path.join(save_dir + '/' + name[0].split('/')[0], 'original_label.nii.gz')
            os.rename(old_name,new_name)

            old_name = os.path.join(save_dir + '/' + name[0].split('/')[0], name_img[0].split('/')[-1] + '_pseudo_label.nii.gz')
            new_name = os.path.join(save_dir + '/' + name[0].split('/')[0], 'pseudo_label.nii.gz')
            os.rename(old_name,new_name)

            destination = save_dir + '/' + name[0].split('/')[0]+'/'+ name_img[0].split('/')[-1]+'/ct.nii.gz'
            try:
                shutil.copy(image_file_path, destination)
                print("Image File copied successfully.")
            except:
                print("Error occurred while copying file.")
            ## load data
            # data = np.load('/out/epoch_80/predict/****.npz')
            # pred, label = data['pred'], data['label']
            lung_data_path = save_dir + '/' + name[0].split('/')[0]+'/'+ name_img[0].split('/')[-1]+'/segmentations/lung_right.nii.gz'
            organ_name=['lung_right','lung_left']
            ct_data = nib.load(destination).get_fdata()
            lung_data = nib.load(lung_data_path).get_fdata()
            lung_data_sum = np.sum(lung_data,axis=(0,1))
            non_zero_idx = np.nonzero(lung_data_sum)
            first_non_zero_idx = non_zero_idx[0][0]
            if total_anomly_slice_number!=0:
                for s in range(first_non_zero_idx,lung_data.shape[-1]):
                    if len(np.unique(ct_data[:,:,s]))!= 1 and lung_data_sum[s] ==0:
                        print('start writing csv as slice: '+str(s+1))
                        with open(save_dir + '/' + name[0].split('/')[0]+'/anomaly.csv','a',newline='') as f:
                            writer = csv.writer(f)
                            content = name_img[0].split('/')[-1]
                            writer.writerow([content,organ_name[0],s+1])
                            writer.writerow([content,organ_name[1],s+1])

            # for s in range(lung_data.shape[-1]):
            #     if len(np.unique(ct_data[:,:,s]))!= 1 and lung_data_sum[s] ==0:
            #         if s==0 and lung_data_sum[s+1]>10000:
            #             print('start writing csv as slice: '+str(s+1))
            #             with open(save_dir + '/' + name[0].split('/')[0]+'/anomaly.csv','a',newline='') as f:
            #                 writer = csv.writer(f)
            #                 content = name_img[0].split('/')[-1]
            #                 writer.writerow([content,organ_name[0],s+1])
            #                 writer.writerow([content,organ_name[1],s+1])
            #         elif s!=0 and s!=lung_data.shape[-1]-1:
            #             if lung_data_sum[s-1]>10000 or lung_data_sum[s+1]>10000:
            #                 print('start writing csv as slice: '+str(s+1))
            #                 with open(save_dir + '/' + name[0].split('/')[0]+'/anomaly.csv','a',newline='') as f:
            #                     writer = csv.writer(f)
            #                     content = name_img[0].split('/')[-1]
            #                     writer.writerow([content,organ_name[0],s+1])
            #                     writer.writerow([content,organ_name[1],s+1])
            #         elif s==lung_data.shape[-1]-1 and lung_data_sum[s-1]>10000:
            #             print('start writing csv as slice: '+str(s+1))
            #             with open(save_dir + '/' + name[0].split('/')[0]+'/anomaly.csv','a',newline='') as f:
            #                 writer = csv.writer(f)
            #                 content = name_img[0].split('/')[-1]
            #                 writer.writerow([content,organ_name[0],s+1])
            #                 writer.writerow([content,organ_name[1],s+1])

            
        torch.cuda.empty_cache()
    
    ave_organ_dice = np.zeros((2, NUM_CLASS))

    # with open('out/'+args.log_name+f'/test_{args.epoch}.txt', 'w') as f:
    #     for key in TEMPLATE.keys():
    #         organ_list = TEMPLATE[key]
    #         content = 'Task%s| '%(key)
    #         for organ in organ_list:
    #             dice = dice_list[key][0][organ-1] / dice_list[key][1][organ-1]
    #             content += '%s: %.4f, '%(ORGAN_NAME[organ-1], dice)
    #             ave_organ_dice[0][organ-1] += dice_list[key][0][organ-1]
    #             ave_organ_dice[1][organ-1] += dice_list[key][1][organ-1]
    #         print(content)
    #         f.write(content)
    #         f.write('\n')
    #     content = 'Average | '
    #     for i in range(NUM_CLASS):
    #         content += '%s: %.4f, '%(ORGAN_NAME[i], ave_organ_dice[0][i] / ave_organ_dice[1][i])
    #     print(content)
    #     f.write(content)
    #     f.write('\n')
    #     print(np.mean(ave_organ_dice[0] / ave_organ_dice[1]))
    #     f.write('%s: %.4f, '%('average', np.mean(ave_organ_dice[0] / ave_organ_dice[1])))
    #     f.write('\n')
        
    
    # np.save(save_dir + '/result.npy', dice_list)
    # load
    # dice_list = np.load(/out/epoch_xxx/result.npy, allow_pickle=True)




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
