# AbdonmenAtlas-8K

We have created a large multi-organ dataset (called AbdomenAtlas-8K) with the spleen, liver, kidneys, stomach, gallbladder, pancreas, aorta, and IVC annotated in **8,448** CT volumes, equating to **3.2 million** CT slices. The conventional annotation methods would take an experienced annotator up to **1,600 weeks** (or roughly **30.8 years**) to complete this task. In contrast, our annotation method has accomplished this task in **three weeks** (based on an 8-hour workday, five days a week) while maintaining a similar or even better annotation quality.

<p align="center"><img width="100%" src="document/fig_dataset_overview.jpg" /></p>

## Paper @Tiezheng

<b>Label-Free Liver Tumor Segmentation</b> <br/>
[Qixin Hu](https://scholar.google.com/citations?user=EqD5GP8AAAAJ&hl=en)<sup>1</sup>, [Yixiong Chen](https://scholar.google.com/citations?hl=en&user=bVHYVXQAAAAJ)<sup>2</sup>, [Junfei Xiao](https://lambert-x.github.io/)<sup>3</sup>, Shuwen Sun<sup>4</sup>, [Jieneng Chen](https://scholar.google.com/citations?hl=en&user=yLYj88sAAAAJ)<sup>3</sup>, [Alan L. Yuille](https://www.cs.jhu.edu/~ayuille/)<sup>3</sup>, and [Zongwei Zhou](https://www.zongweiz.com/)<sup>3,*</sup> <br/>
<sup>1 </sup>Huazhong University of Science and Technology,  <br/>
<sup>2 </sup>The Chinese University of Hong Kong -- Shenzhen,  <br/>
<sup>3 </sup>Johns Hopkins University,   <br/>
<sup>4 </sup>The First Affiliated Hospital of Nanjing Medical University <br/>
arXiv <br/>
[paper](https://arxiv.org/pdf/2303.14869.pdf) | [code](https://github.com/MrGiovanni/SyntheticTumors) | dataset

## 0. Installation

To create environment, see [installation instructions](INSTALL.md).
```bash
git clone https://github.com/MrGiovanni/AbdomenAtlas
cd AbdomenAtlas/pretrained_checkpoints
wget https://www.dropbox.com/s/lyunaue0wwhmv5w/unet.pth
wget https://www.dropbox.com/s/jdsodw2vemsy8sz/swinunetr.pth
cd ..
datapath=/medical_backup/PublicAbdominalData
savepath=/medical_backup/Users/zzhou82/outs
source activate atlas
```


## 1. Download public datasets (BTCV as an example)

```bash
wget https://www.dropbox.com/s/jnv74utwh99ikus/01_Multi-Atlas_Labeling.tar.gz # 01 Multi-Atlas_Labeling.tar.gz (1.53 GB)
tar -xzvf 01_Multi-Atlas_Labeling.tar.gz
```

## 2. Generate a dataset list

```bash
# if the annotation is available
python -W ignore generate_datalist.py --data_path $datapath --dataset_name 01_Multi-Atlas_Labeling --folder img label --out ./dataset/dataset_list --save_file PAOT_01_wt_label.txt

# if the annotation is not available
python -W ignore generate_datalist.py --data_path $datapath --dataset_name 01_Multi-Atlas_Labeling --folder img --out ./dataset/dataset_list --save_file PAOT_01_wo_label.txt
```

## 3. Make AI predictions

###### U-Net
```bash
# if the annotation is available
CUDA_VISIBLE_DEVICES=0 python -W ignore test.py --resume pretrained_checkpoints/unet.pth --backbone unet --save_dir $savepath --dataset_list PAOT_01_wt_label --data_root_path $datapath --original_label  --store_entropy --store_soft_pred --store_result >> logs/PAOT_01_wt_label_unet.txt

# if the annotation is not available
CUDA_VISIBLE_DEVICES=1 python -W ignore test.py --resume pretrained_checkpoints/unet.pth --backbone unet --save_dir $savepath --dataset_list PAOT_01_wo_label --data_root_path $datapath  --store_entropy --store_soft_pred --store_result >> logs/PAOT_01_wo_label_unet.txt
```

###### Swin UNETR
```bash
# if the annotation is available
CUDA_VISIBLE_DEVICES=2 python -W ignore test.py --resume pretrained_checkpoints/swinunetr.pth --backbone swinunetr --save_dir $savepath --dataset_list PAOT_01_wt_label --data_root_path $datapath --original_label  --store_entropy --store_soft_pred --store_result >> logs/PAOT_01_wt_label_swinunetr.txt

# if the annotation is not available
CUDA_VISIBLE_DEVICES=3 python -W ignore test.py --resume pretrained_checkpoints/swinunetr.pth --backbone swinunetr --save_dir $savepath --dataset_list PAOT_01_wo_label --data_root_path $datapath  --store_entropy --store_soft_pred --store_result >> logs/PAOT_01_wo_label_swinunetr.txt
```


## 3. Post-Processing and Generate Dataset @Chongyu ???
```bash
python -W ignore create_dataset.py --dataset_list PAOT_02 PAOT_002 --data_root_path /ccvl/net/ccvl15/chongyu/LargePseudoDataset --save_dir /ccvl/net/ccvl15/chongyu/LargePseudoDataset --model_list unet swinunetr --create_dataset --cpu >> /home/chongyu/tmp/average_02.txt
```

## 4. Generate Attention Map and Priority List @Chongyu ???
```bash
python -W ignore create_attention.py --dataset_list PAOT_02 PAOT_002 --data_root_path /ccvl/net/ccvl15/chongyu/LargePseudoDataset --model_list unet swinunetr --save_consistency --save_entropy --save_overlap >> /home/chongyu/tmp/priority_02.txt
```

<p align="center"><img width="100%" src="document/fig_attention_map.jpg" /></p>
Figure. Illustration of an attention map.

## Citation @Tiezheng

```
@article{hu2023label,
  title={Label-Free Liver Tumor Segmentation},
  author={Hu, Qixin and Chen, Yixiong and Xiao, Junfei and Sun, Shuwen and Chen, Jieneng and Yuille, Alan and Zhou, Zongwei},
  journal={arXiv preprint arXiv:2303.14869},
  year={2023}
}
```

## Acknowledgements
This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research and partially by the Patrick J. McGovern Foundation Award. We appreciate the effort of the MONAI Team to provide open-source code for the community.
