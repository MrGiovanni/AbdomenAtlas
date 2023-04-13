#This is for generating the datalist of files end with .nii.gz
import argparse
from pathlib import Path
import os


def generate_list(args):
    data_path = Path(args.data_path)
    dataset_path = data_path.joinpath(args.dataset_name)
    files = sorted(list(dataset_path.rglob('*' + args.filetype)))
    files = [str(file.relative_to(data_path)) for file in files if not file.name.startswith('.')]
    if args.folder:
        files = [file for file in files if any(folder in file for folder in args.folder)]
    if len(args.folder) > 1:
        part1 = []
        part2 = []
        for f in files:
            if args.folder[0] in f:
                part1.append(f)
            if args.folder[1] in f:
                part2.append(f)
        if len(part1) == len(part2):
            files = [part1[i] + '\t'+part2[i] for i in range(len(part1))]
        else:
            print('Please check the number of your images and labels')
        # files = [files[i] + '\t' + files[i + len(files) // 2] for i in range(len(files) // 2)]
    
    with open(os.path.join(args.out, args.save_file), 'w') as f:
        f.write('\n'.join(files))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/mnt/zzhou82/PublicAbdominalData', help='The path of your data')
    parser.add_argument('--dataset_name', default='16_TotalSegmentor', help='The dataset name for generating')
    parser.add_argument('--folder', nargs='+', default=None, help='folder to filter the files(img,train,imagesTr)')
    parser.add_argument('--filetype', default='.nii.gz', help='.nii.gz,.mhd')
    parser.add_argument('--out', default='./dataset/dataset_list')
    parser.add_argument('--save_file', default='PAOT_18_wt_label.txt')

    args = parser.parse_args()

    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    generate_list(args)


if __name__ == "__main__":
    main()
