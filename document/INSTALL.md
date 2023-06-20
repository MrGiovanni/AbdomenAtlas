# Installation
##### Create Environments
```bash
module load anaconda3/5.3.0 # only for Agave
module load mamba/latest # only for Sol
mamba create -n atlas python=3.9

conda create -n atlas python=3.9
source activate atlas
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install monai[all]==0.9.0
pip install -r requirements.txt
```
##### Download Pretrained Weights
```bash
git clone https://github.com/MrGiovanni/AbdomenAtlas
cd AbdomenAtlas/pretrained_checkpoints
wget swinunetr.pth (coming soon)
wget unet.pth (coming soon)
cd ..
cd pretrained_weights/
wget https://www.dropbox.com/s/po2zvqylwr0fuek/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt
cd ../
dataname=01_Multi-Atlas_Labeling # an example
datapath=/medical_backup/PublicAbdominalData/
savepath=/medical_backup/Users/zzhou82/outs
source activate atlas
```