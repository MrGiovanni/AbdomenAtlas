#This is for generating the datalist of files end with .nii.gz
import argparse
from pathlib import Path
import os


def generate_list(args):
    data_path = Path(args.data_path)
    dataset_path = data_path.joinpath(args.dataset_name)
    files = sorted(list(dataset_path.rglob('*' + args.filetype)))
    files = [str(file.relative_to(data_path)) for file in files if not file.name.startswith('.')]
    if args.flag:
        files = [file for file in files if any(flag in file for flag in args.flag)]
    if len(args.flag) > 1:
        part1 = []
        part2 = []
        for f in files:
            if args.flag[0] in f:
                part1.append(f)
            if args.flag[1] in f:
                part2.append(f)
        if len(part1) == len(part2):
            files = [part1[i] + '\t'+part2[i] for i in range(len(part1))]
        else:
            print('Please check the number of your images and labels')
        # files = [files[i] + '\t' + files[i + len(files) // 2] for i in range(len(files) // 2)]
    if len(args.flag)>1:
        with open(Path(args.out).joinpath(f"PAOT_{args.dataset_name}_withlabel_test.txt"), 'w') as f:
            f.write('\n'.join(files)) 
    else:
        with open(Path(args.out).joinpath(f"PAOT_{args.dataset_name}_withoutlabel_test.txt"), 'w') as f:
            f.write('\n'.join(files))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/mnt/zzhou82/PublicAbdominalData', help='The path of your data')
    parser.add_argument('--dataset_name', default='16_TotalSegmentor', help='The dataset name for generating')
    parser.add_argument('--flag', nargs='+', default=None, help='Flag to filter the files(img,train,imagesTr)')
    parser.add_argument('--filetype', default='.nii.gz', help='.nii.gz,.mhd')
    parser.add_argument('--out', default='./dataset/dataset_list')

    args = parser.parse_args()

    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    generate_list(args)


if __name__ == "__main__":
    main()
