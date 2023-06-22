# Installation
### Create Environments
```bash
conda create -n atlas python=3.9
source activate atlas
cd AbdomenAtlas/
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install monai[all]==0.9.0
pip install -r requirements.txt
```

##### [Optional] If You are using ASU GPU Cluster

Please first read the document at [Essence for Linux newbies](https://github.com/MrGiovanni/Eureka/blob/master/Essence%20for%20Linux%20newbies.md#access-asu-gpu-cluster)
```bash
module load anaconda3/5.3.0 # only for Agave

module load mamba/latest # only for Sol
mamba create -n atlas python=3.9
```

### Download Pretrained Weights

```bash
cd pretrained_weights/
wget https://www.dropbox.com/s/po2zvqylwr0fuek/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt
wget https://www.dropbox.com/s/lh5kuyjxwjsxjpl/Genesis_Chest_CT.pt
cd ../
cd pretrained_checkpoints/
wget swinunetr.pth (coming soon)
wget unet.pth (coming soon)
cd ..
```

### Define Variables

```bash
dataname=01_Multi-Atlas_Labeling # an example
datapath=/medical_backup/PublicAbdominalData/
savepath=/medical_backup/Users/zzhou82/outs
```
