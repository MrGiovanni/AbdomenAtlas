# LargePseudoDataset

```
source /data/zzhou82/environments/universal/bin/activate
cd /data/zzhou82/project/4Feb2023_LargePseudoDataset/
CUDA_VISIBLE_DEVICES=1 python -W ignore test.py --resume /data/zzhou82/project/4Feb2023_LargePseudoDataset/pretrained_checkpoints/epoch_450.pth --log_name /data/zzhou82/dataset/LargePseudoDataset --dataset_list PAOT_01 --data_root_path /mnt/zzhou82/PublicAbdominalData/ --store_result  >> /data/zzhou82/project/4Feb2023_LargePseudoDataset/logs/PAOT_01.txt

source /data/zzhou82/environments/universal/bin/activate
cd /data/zzhou82/project/4Feb2023_LargePseudoDataset/
CUDA_VISIBLE_DEVICES=1 python -W ignore test.py --resume /data/zzhou82/project/4Feb2023_LargePseudoDataset/pretrained_checkpoints/epoch_450.pth --log_name /data/zzhou82/dataset/LargePseudoDataset --dataset_list PAOT_02 --data_root_path /mnt/zzhou82/PublicAbdominalData/ --store_result  >> /data/zzhou82/project/4Feb2023_LargePseudoDataset/logs/PAOT_02.txt

source /data/zzhou82/environments/universal/bin/activate
cd /data/zzhou82/project/4Feb2023_LargePseudoDataset/
CUDA_VISIBLE_DEVICES=3 python -W ignore test.py --resume /data/zzhou82/project/4Feb2023_LargePseudoDataset/pretrained_checkpoints/epoch_450.pth --log_name /data/zzhou82/dataset/LargePseudoDataset --dataset_list PAOT_03 --data_root_path /mnt/zzhou82/PublicAbdominalData/ --store_result  >> /data/zzhou82/project/4Feb2023_LargePseudoDataset/logs/PAOT_03.txt

source /data/zzhou82/environments/universal/bin/activate
cd /data/zzhou82/project/4Feb2023_LargePseudoDataset/
CUDA_VISIBLE_DEVICES=6 python -W ignore test.py --resume /data/zzhou82/project/4Feb2023_LargePseudoDataset/pretrained_checkpoints/epoch_450.pth --log_name /data/zzhou82/dataset/LargePseudoDataset --dataset_list PAOT_05 --data_root_path /mnt/zzhou82/PublicAbdominalData/ --store_result  >> /data/zzhou82/project/4Feb2023_LargePseudoDataset/logs/PAOT_05.txt

source /data/zzhou82/environments/universal/bin/activate
cd /data/zzhou82/project/4Feb2023_LargePseudoDataset/
CUDA_VISIBLE_DEVICES=5 python -W ignore test.py --resume /data/zzhou82/project/4Feb2023_LargePseudoDataset/pretrained_checkpoints/epoch_450.pth --log_name /data/zzhou82/dataset/LargePseudoDataset --dataset_list PAOT_07 --data_root_path /mnt/zzhou82/PublicAbdominalData/ --store_result  >> /data/zzhou82/project/4Feb2023_LargePseudoDataset/logs/PAOT_07.txt

source /data/zzhou82/environments/universal/bin/activate
cd /data/zzhou82/project/4Feb2023_LargePseudoDataset/
CUDA_VISIBLE_DEVICES=0 python -W ignore test.py --resume /data/zzhou82/project/4Feb2023_LargePseudoDataset/pretrained_checkpoints/epoch_450.pth --log_name /data/zzhou82/dataset/LargePseudoDataset --dataset_list PAOT_14 --data_root_path /mnt/zzhou82/PublicAbdominalData/ --store_result  >> /data/zzhou82/project/4Feb2023_LargePseudoDataset/logs/PAOT_14.txt
```
