# Installation

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
