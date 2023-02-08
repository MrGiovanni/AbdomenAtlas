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
pip install --upgrade monai==0.9.0
pip install -r requirements.txt
cd pretrained_checkpoints
wget https://www.dropbox.com/s/6ggd0gq5qddahwh/epoch_450.pth
cd ..
```

#### Generate pseudo labels for different datasets

```
source /data/zzhou82/environments/universal/bin/activate
cd /data/zzhou82/project/4Feb2023_LargePseudoDataset/
CUDA_VISIBLE_DEVICES=1 python -W ignore test.py --resume pretrained_checkpoints/epoch_450.pth --log_name /mnt/zzhou82/LargePseudoDataset --dataset_list PAOT_01 --data_root_path /mnt/zzhou82/PublicAbdominalData/ --store_result  >> logs/PAOT_01.txt

source /data/zzhou82/environments/universal/bin/activate
cd /data/zzhou82/project/4Feb2023_LargePseudoDataset/
CUDA_VISIBLE_DEVICES=1 python -W ignore test.py --resume pretrained_checkpoints/epoch_450.pth --log_name /mnt/zzhou82/LargePseudoDataset --dataset_list PAOT_02 --data_root_path /mnt/zzhou82/PublicAbdominalData/ --store_result  >> logs/PAOT_02.txt

source /data/zzhou82/environments/universal/bin/activate
cd /data/zzhou82/project/4Feb2023_LargePseudoDataset/
CUDA_VISIBLE_DEVICES=3 python -W ignore test.py --resume pretrained_checkpoints/epoch_450.pth --log_name /mnt/zzhou82/LargePseudoDataset --dataset_list PAOT_03 --data_root_path /mnt/zzhou82/PublicAbdominalData/ --store_result  >> logs/PAOT_03.txt

source /data/zzhou82/environments/universal/bin/activate
cd /data/zzhou82/project/4Feb2023_LargePseudoDataset/
CUDA_VISIBLE_DEVICES=6 python -W ignore test.py --resume pretrained_checkpoints/epoch_450.pth --log_name /mnt/zzhou82/LargePseudoDataset --dataset_list PAOT_04 --data_root_path /mnt/zzhou82/PublicAbdominalData/ --store_result  >> logs/PAOT_04.txt

source /data/zzhou82/environments/universal/bin/activate
cd /data/zzhou82/project/4Feb2023_LargePseudoDataset/
CUDA_VISIBLE_DEVICES=2 python -W ignore test.py --resume pretrained_checkpoints/epoch_450.pth --log_name /mnt/zzhou82/LargePseudoDataset --dataset_list PAOT_05 --data_root_path /mnt/zzhou82/PublicAbdominalData/ --store_result  >> logs/PAOT_05.txt

source /data/zzhou82/environments/universal/bin/activate
cd /data/zzhou82/project/4Feb2023_LargePseudoDataset/
CUDA_VISIBLE_DEVICES=5 python -W ignore test.py --resume pretrained_checkpoints/epoch_450.pth --log_name /mnt/zzhou82/LargePseudoDataset --dataset_list PAOT_07 --data_root_path /mnt/zzhou82/PublicAbdominalData/ --store_result  >> logs/PAOT_07.txt

source /data/zzhou82/environments/universal/bin/activate
cd /data/zzhou82/project/4Feb2023_LargePseudoDataset/
CUDA_VISIBLE_DEVICES=0 python -W ignore test.py --resume pretrained_checkpoints/epoch_450.pth --log_name /mnt/zzhou82/LargePseudoDataset --dataset_list PAOT_08 --data_root_path /mnt/zzhou82/PublicAbdominalData/ --store_result  >> logs/PAOT_08.txt

source /data/zzhou82/environments/universal/bin/activate
cd /data/zzhou82/project/4Feb2023_LargePseudoDataset/
CUDA_VISIBLE_DEVICES=1 python -W ignore test.py --resume pretrained_checkpoints/epoch_450.pth --log_name /mnt/zzhou82/LargePseudoDataset --dataset_list PAOT_09 --data_root_path /mnt/zzhou82/PublicAbdominalData/ --store_result  >> logs/PAOT_09.txt

source /data/zzhou82/environments/universal/bin/activate
cd /data/zzhou82/project/4Feb2023_LargePseudoDataset/
CUDA_VISIBLE_DEVICES=3 python -W ignore test.py --resume pretrained_checkpoints/epoch_450.pth --log_name /mnt/zzhou82/LargePseudoDataset --dataset_list PAOT_10 --data_root_path /mnt/zzhou82/PublicAbdominalData/ --store_result  >> logs/PAOT_10.txt

source /data/zzhou82/environments/universal/bin/activate
cd /data/zzhou82/project/4Feb2023_LargePseudoDataset/
CUDA_VISIBLE_DEVICES=4 python -W ignore test.py --resume pretrained_checkpoints/epoch_450.pth --log_name /mnt/zzhou82/LargePseudoDataset --dataset_list PAOT_12 --data_root_path /mnt/zzhou82/PublicAbdominalData/ --store_result  >> logs/PAOT_12.txt

source /data/zzhou82/environments/universal/bin/activate
cd /data/zzhou82/project/4Feb2023_LargePseudoDataset/
CUDA_VISIBLE_DEVICES=3 python -W ignore test.py --resume pretrained_checkpoints/epoch_450.pth --log_name /mnt/zzhou82/LargePseudoDataset --dataset_list PAOT_13 --data_root_path /mnt/zzhou82/PublicAbdominalData/ --store_result  >> logs/PAOT_13.txt

source /data/zzhou82/environments/universal/bin/activate
cd /data/zzhou82/project/4Feb2023_LargePseudoDataset/
CUDA_VISIBLE_DEVICES=0 python -W ignore test.py --resume pretrained_checkpoints/epoch_450.pth --log_name /mnt/zzhou82/LargePseudoDataset --dataset_list PAOT_14 --data_root_path /mnt/zzhou82/PublicAbdominalData/ --store_result  >> logs/PAOT_14.txt
```

#### Experiment logs

- [x] PAOT_01
- [x] PAOT_02
- [x] PAOT_03
- [x] PAOT_04
- [ ] PAOT_05
- [x] PAOT_07
- [ ] PAOT_08
- [ ] PAOT_09
- [ ] PAOT_10
- [ ] PAOT_12
- [x] PAOT_13
- [ ] PAOT_14
