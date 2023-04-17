# Large Pseudo Dataset

#### Setup
```
cd /data/zzhou82/environments
python3 -m venv universal
source /data/zzhou82/environments/universal/bin/activate

cd /data/zzhou82/project
git clone https://github.com/MrGiovanni/AbdomenAtlas
cd LargePseudoDataset
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install 'monai[all]'
pip install --upgrade monai==0.9.0
pip install -r requirements.txt
cd pretrained_checkpoints
wget https://www.dropbox.com/s/jdsodw2vemsy8sz/swinunetr.pth
wget https://www.dropbox.com/s/lyunaue0wwhmv5w/unet.pth
cd ..
```

#### Download public datasets

```tar -xzvf name.tar.gz```

```
wget https://www.dropbox.com/s/jnv74utwh99ikus/01_Multi-Atlas_Labeling.tar.gz # 01 Multi-Atlas_Labeling.tar.gz (1.53 GB)
wget https://www.dropbox.com/s/5yzdzb7el9r3o9i/02_TCIA_Pancreas-CT.tar.gz # 02 TCIA_Pancreas-CT.tar.gz (7.51 GB)
wget https://www.dropbox.com/s/lzrhirei2t2vuwg/03_CHAOS.tar.gz # 03 CHAOS.tar.gz (925.3 MB)
wget https://www.dropbox.com/s/2i19kuw7qewzo6q/04_LiTS.tar.gz # 04 LiTS.tar.gz (17.42 GB)
wget https://www.dropbox.com/s/l2bvis2pjlcyas7/05_KiTS.tar.gz # 05 KiTS.tar.gz (28.04 GB)
wget https://www.dropbox.com/s/toavg919niykblq/07_WORD.tar.gz # 07 WORD.tar.gz (5.31 GB)
wget https://www.dropbox.com/s/70e3df92w3imggh/08_AbdomenCT-1K.tar.gz # 08 AbdomenCT-1K.tar.gz (82.54 GB)
wget https://www.dropbox.com/s/7ro2nsmhf1cq2xn/09_AMOS.tar.gz # 09 AMOS.tar.gz (8.81 GB)
wget https://www.dropbox.com/s/e7yq57esg3sci3m/10_Decathlon.tar.gz # 10 Decathlon.tar.gz (75.31 GB)
wget https://www.dropbox.com/s/x6slst6kt9pdg2t/12_CT-ORG.tar.gz # 12 CT-ORG.tar.gz (18.03 GB)
wget https://www.dropbox.com/s/6vp6o8tydb8waby/13_AbdomenCT-12organ.tar.gz # 13 AbdomenCT-12organ.tar.gz (1.48 GB)
wget https://www.dropbox.com/s/ipkeaelyethy3sn/Totalsegmentator_dataset.zip # Totalsegmentor
```

#### 1. Generate dataset list
```bash
python -W ignore generate_datalist.py --data_path /medical_backup/PublicAbdominalData --dataset_name 18_FLARE23 --folder imagesTr2200 labelsTr2200 --out ./dataset/dataset_list --save_file PAOT_18_wt_label.txt

python -W ignore generate_datalist.py --data_path /medical_backup/PublicAbdominalData --dataset_name 18_FLARE23 --folder unlabeledTr1800 --out ./dataset/dataset_list --save_file PAOT_18_wo_label.txt
```

#### 2. Make AI predictions
```bash
source /data/zzhou82/environments/universal/bin/activate
cd /data/zzhou82/project/4Feb2023_LargePseudoDataset/
CUDA_VISIBLE_DEVICES=0 python -W ignore test.py --resume pretrained_checkpoints/unet.pth --backbone unet --save_dir /data/zzhou82/project/LargePseudoDataset/outs --dataset_list PAOT_18_wt_label --data_root_path /medical_backup/PublicAbdominalData/ --original_label  --store_entropy --store_soft_pred --store_result >> logs/PAOT_18_wt_label_unet.txt
CUDA_VISIBLE_DEVICES=1 python -W ignore test.py --resume pretrained_checkpoints/unet.pth --backbone unet --save_dir /data/zzhou82/project/LargePseudoDataset/outs --dataset_list PAOT_18_wo_label --data_root_path /medical_backup/PublicAbdominalData/  --store_entropy --store_soft_pred --store_result >> logs/PAOT_18_wo_label_unet.txt

CUDA_VISIBLE_DEVICES=2 python -W ignore test.py --resume pretrained_checkpoints/swinunetr.pth --backbone swinunetr --save_dir /data/zzhou82/project/LargePseudoDataset/outs --dataset_list PAOT_18_wt_label --data_root_path /medical_backup/PublicAbdominalData/ --original_label  --store_entropy --store_soft_pred --store_result >> logs/PAOT_18_wt_label_swinunetr.txt
CUDA_VISIBLE_DEVICES=3 python -W ignore test.py --resume pretrained_checkpoints/swinunetr.pth --backbone swinunetr --save_dir /data/zzhou82/project/LargePseudoDataset/outs --dataset_list PAOT_18_wo_label --data_root_path /medical_backup/PublicAbdominalData/  --store_entropy --store_soft_pred --store_result >> logs/PAOT_18_wo_label_swinunetr.txt
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
CUDA_VISIBLE_DEVICES=0 python -W ignore test.py --resume pretrained_checkpoints/epoch_450.pth --log_name /mnt/zzhou82/LargePseudoDataset --dataset_list PAOT_12 --data_root_path /mnt/zzhou82/PublicAbdominalData/ --store_result  >> logs/PAOT_12.txt

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
- [x] PAOT_05
- [x] PAOT_07
- [x] PAOT_08
- [x] PAOT_09
- [x] PAOT_10
- [x] PAOT_12
- [x] PAOT_13
- [ ] PAOT_14
