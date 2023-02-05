# LargePseudoDataset

#### Setup
```
cd /data/zzhou82/environments
python3 -m venv universal
source /data/zzhou82/environments/universal/bin/activate

cd /data/zzhou82/project
git clone https://github.com/MrGiovanni/LargePseudoDataset.git
cd LargePseudoDataset
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install 'monai[all]'
pip install -r requirements.txt
cd pretrained_checkpoints
wget https://www.dropbox.com/s/6ggd0gq5qddahwh/epoch_450.pth
cd ..
```

#### Generate pseudo labels for different datasets

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
