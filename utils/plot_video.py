# """
# source /data/zzhou82/environments/syn/bin/activate
# python -W ignore plot_video.py --abdomen_atlas /mnt/tzhang85/AbdomenAtlas_8K_internal --start_index 4000 --end_index 4420
# """

import numpy as np 
import os 
import cv2
import random
import argparse
import nibabel as nib 
from tqdm import tqdm 
from PIL import Image
import imageio
from scipy.ndimage import label

parser = argparse.ArgumentParser()
parser.add_argument('--abdomen_atlas', dest='abdomen_atlas', type=str, default='/Volumes/Atlas/AbdomenAtlas_8K_internal',
                    help='the directory of the AbdomenAtlas dataset',
                   )
parser.add_argument("--png_save_path", dest='png_save_path', type=str, default='./materials',
                    help='the directory of png for each CT slice',
                   )
parser.add_argument("--video_save_path", dest='video_save_path', type=str, default='./videos',
                    help='the directory for saving videos',
                   )
parser.add_argument("--gif_save_path", dest='gif_save_path', type=str, default='./gifs',
                    help='the directory for saving gifs',
                   )
parser.add_argument("--FPS", dest='FPS', type=float, default=20,
                    help='the FPS value for videos',
                   )
parser.add_argument("--start_index", dest='start_index', type=int, default=0,
                    help='the start index of CT volumes',
                   )
parser.add_argument("--end_index", dest='end_index', type=int, default=100,
                    help='the end index of CT volumes',
                   )
args = parser.parse_args()

if not os.path.exists(args.png_save_path):
    os.makedirs(args.png_save_path)
    
if not os.path.exists(args.video_save_path):
    os.makedirs(args.video_save_path)
    
if not os.path.exists(args.gif_save_path):
    os.makedirs(args.gif_save_path)

low_range = -200
high_range = 400

CLASS_IND = {
    'spleen': 1,
    'kidney R': 2,
    'kidney L': 3,
    'gall bladder': 4,
    'esophagus': 5,
    'liver': 6,
    'stomach': 7,
    'aorta': 8,
    'postcava': 9,
    'portal and splenic vein': 10,
    'pancreas': 11,
    'adrenal gland R': 12,
    'adrenal gland L': 13,
    'duodenum': 14,
    'hepatic vessel': 15,
    'lung R': 16,
    'lung L': 17,
    'colon': 18,
    'intestine': 19,
    'rectum': 20,
    'bladder': 21,
    'prostate': 22,
    'head of femur L': 23,
    'head of femur R': 24,
    'celiac trunk': 25,
    'kidney tumor': 26,
    'liver tumor': 27,
    'pancreatic tumor': 28,
    'hepatic vessel tumor': 29,
    'lung tumor': 30,
    'colon tumor': 31,
    'kidney cyst': 32,
}

def add_colorful_mask(image, mask, class_index):
    
    image[mask == class_index['spleen'], 0] = 255   # spleen (255,0,0) 
    
    image[mask == class_index['kidney R'], 1] = 255   # kidney R (0,255,0)
    image[mask == class_index['kidney L'], 1] = 255   # kidney L (0,255,0)
    image[mask == class_index['kidney tumor'], 1] = 255  # kidney tumor (0,255,0)
    image[mask == class_index['kidney cyst'], 1] = 255  # kidney cyst (0,255,0)
    
    image[mask == class_index['gall bladder'], 0] = 255   # gall bladder (255,255,0)
    image[mask == class_index['gall bladder'], 1] = 255   # 
    
    # image[mask == class_index['esophagus'], 1] = 255   # esophagus (0,255,255)
    # image[mask == class_index['esophagus'], 2] = 255   # 
    image[mask == class_index['liver'], 0] = 255   # liver (255,0,255)
    image[mask == class_index['hepatic vessel'], 0] = 255  # hepatic vessel (255,0,255)
    image[mask == class_index['liver tumor'], 0] = 255  # liver tumors (255,0,255)
    image[mask == class_index['hepatic vessel tumor'], 0] = 255  # hepatic vessel tumors (255,0,255)
    image[mask == class_index['liver'], 2] = 255   # liver (255,0,255)
    image[mask == class_index['hepatic vessel'], 2] = 255  # hepatic vessel (255,0,255)
    image[mask == class_index['liver tumor'], 2] = 255  # liver tumors (255,0,255)
    image[mask == class_index['hepatic vessel tumor'], 2] = 255  # hepatic vessel tumors (255,0,255)
    
    image[mask == class_index['stomach'], 0] = 255
    image[mask == class_index['stomach'], 1] = 239   # stomach (255,239,255)
    image[mask == class_index['stomach'], 2] = 213   # 
    
    image[mask == class_index['aorta'], 1] = 255
    image[mask == class_index['aorta'], 2] = 255   # aorta (0,255,255)
    
    image[mask == class_index['postcava'], 0] = 205   # postcava (205,133,63)
    image[mask == class_index['postcava'], 1] = 133   # 
    image[mask == class_index['postcava'], 2] = 63 # + image[mask == class_index['postcava'], 2] * 0.2   # 
    
    # image[mask == class_index['portal and splenic vein'], 0] = 0 + image[mask == class_index['portal and splenic vein'], 0] * 0.5 # portal and splenic vein (0,0,255)
    # image[mask == class_index['portal and splenic vein'], 1] = 0 + image[mask == class_index['portal and splenic vein'], 1] * 0.5 # 
    # image[mask == class_index['portal and splenic vein'], 2] = 255  # 
    
    image[mask == class_index['pancreas'], 0] = 102  # pancreas (102,205,170)
    image[mask == class_index['pancreas'], 1] = 205
    image[mask == class_index['pancreas'], 2] = 170  #  
    image[mask == class_index['pancreatic tumor'], 0] = 102  # pancreatic tumors (102,205,170)
    image[mask == class_index['pancreatic tumor'], 1] = 205
    image[mask == class_index['pancreatic tumor'], 2] = 170

    # image[mask == class_index['adrenal gland R'], 0] = 0 + image[mask == class_index['adrenal gland R'], 0] * 0.5 # adrenal gland R (0,255,0)
    # image[mask == class_index['adrenal gland R'], 1] = 255 # 
    # image[mask == class_index['adrenal gland R'], 2] = 0 + image[mask == class_index['adrenal gland R'], 2] * 0.5  # 
    # image[mask == class_index['adrenal gland L'], 0] = 0 + image[mask == class_index['adrenal gland L'], 0] * 0.5 # adrenal gland L (0,255,0)
    # image[mask == class_index['adrenal gland L'], 1] = 255 # 
    # image[mask == class_index['adrenal gland L'], 2] = 0 + image[mask == class_index['adrenal gland L'], 2] * 0.5 # 
    
    # image[mask == class_index['duodenum'], 0] = 255 # duodenum (255,80,80)
    # image[mask == class_index['duodenum'], 1] = 80 + image[mask == class_index['duodenum'], 1] * 0.6 # 
    # image[mask == class_index['duodenum'], 2] = 80 + image[mask == class_index['duodenum'], 2] * 0.6 # 
    
    # image[mask == class_index['lung R'], 0] = 200 + image[mask == class_index['lung R'], 0] * 0.2  # lung R (200,128,0)
    # image[mask == class_index['lung R'], 2] = 128 + image[mask == class_index['lung R'], 2] * 0.5  #
    # image[mask == class_index['lung L'], 0] = 200 + image[mask == class_index['lung L'], 0] * 0.2  # lung L (200,128,0)
    # image[mask == class_index['lung L'], 2] = 128 + image[mask == class_index['lung L'], 2] * 0.5  #
    # image[mask == class_index['lung tumor'], 0] = 200 + image[mask == class_index['lung tumor'], 0] * 0.2  # lung tumor (200,128,0)
    # image[mask == class_index['lung tumor'], 2] = 128 + image[mask == class_index['lung tumor'], 2] * 0.5  #
    
    # image[mask == class_index['colon'], 0] = 170  # colon (170,0,255)
    # image[mask == class_index['colon'], 1] = 0 + image[mask == class_index['colon'], 1] * 0.7    # 
    # image[mask == class_index['colon'], 2] = 255  # 
    # image[mask == class_index['colon tumor'], 0] = 170  # colon tumors (170,0,255)
    # image[mask == class_index['colon tumor'], 1] = 0 + image[mask == class_index['colon tumor'], 1] * 0.7    # 
    # image[mask == class_index['colon tumor'], 2] = 255  #
    
    # image[mask == class_index['prostate'], 0] = 0    # prostate (0,128,128)
    # image[mask == class_index['prostate'], 1] = 128  # 
    # image[mask == class_index['prostate'], 2] = 128 + image[mask == class_index['prostate'], 2] * 0.5  # 
    
    # image[mask == class_index['celiac trunk'], 0] = 255    # celiac trunk (255,0,0)
    # image[mask == class_index['celiac trunk'], 1] = 0 + image[mask == class_index['celiac trunk'], 1] * 0.5  # 
    # image[mask == class_index['celiac trunk'], 2] = 0  # 
    
    return image

def make_png(case_name, args):
    
    image_name = f'ct.nii.gz'
    mask_name = f'pseudo_label.nii.gz'

    image_path = os.path.join(args.abdomen_atlas, case_name, image_name)
    mask_path = os.path.join(args.abdomen_atlas, case_name, mask_name)

    # single case
    image = nib.load(image_path).get_fdata().astype(np.int16)
    mask = nib.load(mask_path).get_fdata().astype(np.uint8)
    
    # change orientation
    ornt = nib.orientations.axcodes2ornt(('F', 'L', 'U'), (('L','R'),('B','F'),('D','U')))
    image = nib.orientations.apply_orientation(image, ornt)
    mask = nib.orientations.apply_orientation(mask, ornt)
    
    image[image > high_range] = high_range
    image[image < low_range] = low_range
    image = np.round((image - low_range) / (high_range - low_range) * 255.0).astype(np.uint8)
    image = np.repeat(image.reshape(image.shape[0],image.shape[1],image.shape[2],1), 3, axis=3)
    
    image_mask = add_colorful_mask(image, mask, CLASS_IND)
    
    for plane in ['axial', 'coronal', 'sagittal']:
        if not os.path.exists(os.path.join(args.png_save_path, plane, case_name)):
            os.makedirs(os.path.join(args.png_save_path, plane, case_name))
    
    for z in range(600,801):
        Image.fromarray(np.flip(image_mask[:,:,z,:],axis = 0)).save(os.path.join(args.png_save_path, 'axial', case_name, str(z)+'.png'))

    for z in range(mask.shape[1]):
        Image.fromarray(image_mask[:,z,:,:]).save(os.path.join(args.png_save_path, 'sagittal', case_name, str(z)+'.png'))

    for z in range(mask.shape[0]):
        Image.fromarray(image_mask[z,:,:,:]).save(os.path.join(args.png_save_path, 'coronal', case_name, str(z)+'.png'))

def make_avi(case_name, plane, args):
    
    image_folder = os.path.join(args.png_save_path, plane, case_name)
    video_name = os.path.join(args.video_save_path, plane, case_name+'.avi')
    gif_name = os.path.join(args.gif_save_path, plane, case_name+'.gif')
    
    if not os.path.exists(os.path.join(args.video_save_path, plane)):
        os.makedirs(os.path.join(args.video_save_path, plane))
    if not os.path.exists(os.path.join(args.gif_save_path, plane)):
        os.makedirs(os.path.join(args.gif_save_path, plane))

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    for i in range(len(images)):
        images[i] = images[i].replace('.png','')
        images[i] = int(images[i])
    images.sort()

    frame = cv2.imread(os.path.join(image_folder, str(images[0])+'.png'))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, args.FPS, (width,height))

    imgs = []
    for image in images:
        img = cv2.imread(os.path.join(image_folder, str(image)+'.png'))
        video.write(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

    cv2.destroyAllWindows()
    video.release()
    imageio.mimsave(gif_name, imgs, fps=args.FPS)
        
if __name__ == "__main__":
    
    folder_names = [name for name in os.listdir(args.abdomen_atlas) if os.path.isdir(os.path.join(args.abdomen_atlas, name))]
    for folder in tqdm(folder_names[args.start_index:args.end_index]):
        print(folder)
        if folder == '.ipynb_checkpoints':
            continue
        make_png(folder, args)
        for plane in ['axial', 'coronal', 'sagittal']:
            make_avi(folder, plane, args)