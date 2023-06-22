import numpy as np 
import os 
from PIL import Image
import cv2
import imageio
import argparse


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
parser.add_argument('--volume_list', nargs='+', default=['14_FELIX_ARTERIAL_FELIX1261_ARTERIAL', '14_FELIX_ARTERIAL_FELIX0065_ARTERIAL','14_FELIX_ARTERIAL_FELIX0378_ARTERIAL','14_FELIX_ARTERIAL_FELIX0459_ARTERIAL','14_FELIX_ARTERIAL_FELIX5566_ARTERIAL'])
args = parser.parse_args()




def images_sort(image_folder):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images_path = []
    for i in range(len(images)):
        images[i] = images[i].replace('.png','')
        images[i] = int(images[i])
    images.sort()
    for image in images:
        images_path.append(os.path.join(image_folder,str(image)+'.png'))
    return images_path

def combine_image(images_list):
    images_length = len(images_list[0])
    combine_image_path = os.path.join(args.png_save_path,'axial','combine')
    combine_images = []
    if not os.path.exists(combine_image_path):
        os.makedirs(combine_image_path)
    for index in range(images_length):
        
        combine_images_list = []
        for images in images_list:
            image = Image.open(images[index])
            combine_images_list.append(image)
        
        new_width = len(combine_images_list)*combine_images_list[0].size[0]
        height = combine_images_list[0].size[1]
        new_image = Image.new("RGBA",(new_width,height))
        for i in range(len(combine_images_list)):

            new_image.paste(combine_images_list[i],(i*combine_images_list[0].size[0],0))
        new_image.save(os.path.join(combine_image_path,str(index)+'.png'),format = 'png')
        combine_images.append(new_image)
    imageio.mimsave(os.path.join(args.gif_save_path,'visualize.gif'), combine_images, fps=args.FPS)

if __name__ == "__main__":
    images_list = []
    for item in args.volume_list:
        volume_path = os.path.join(args.png_save_path,'axial',item)
        images_path = images_sort(volume_path)
        images_list.append(images_path)
    combine_image(images_list)

    