# AbdonmenAtlas 1.0

We are proud to introduce AbdomenAtlas-8K, a substantial multi-organ dataset with the spleen, liver, kidneys, stomach, gallbladder, pancreas, aorta, and IVC annotated in **8,448** CT volumes, totaling **3.2 million** CT slices. 

An endeavor of such magnitude would demand a staggering **1,600 weeks** or roughly **30.8 years** of an experienced annotator's time. 

In contrast, our annotation method has accomplished this task in **three weeks** (premised on an 8-hour workday, five days a week) while maintaining a similar or even better annotation quality.

<p align="center"><img width="100%" src="document/fig_dataset_overview.jpg" /></p>
<p align="center"><img width="100%" src="document/fig_dataset.gif" /></p>
<p align="center"><img width="60%" src="document/fig_legend.png" /></p>

## Paper

<b>AbdomenAtlas-8K: Annotating 8,000 CT Volumes for Multi-Organ Segmentation in Three Weeks</b> <br/>
[Chongyu Qu](https://github.com/Chongyu1117)<sup>1</sup>, [Tiezheng Zhang](https://github.com/ollie-ztz)<sup>1</sup>, [Hualin Qiao](https://www.linkedin.com/in/hualin-qiao-a29438210/)<sup>2</sup>, [Jie Liu](https://ljwztc.github.io/)<sup>3</sup>, [Yucheng Tang](https://scholar.google.com/citations?hl=en&user=0xheliUAAAAJ)<sup>4</sup>, [Alan L. Yuille](https://www.cs.jhu.edu/~ayuille/)<sup>1</sup>, and [Zongwei Zhou](https://www.zongweiz.com/)<sup>1,*</sup> <br/>
<sup>1 </sup>Johns Hopkins University,  <br/>
<sup>2 </sup>Rutgers University,  <br/>
<sup>3 </sup>City University of Hong Kong,   <br/>
<sup>4 </sup>NVIDIA <br/>
NeurIPS 2023 <br/>
[paper](https://www.cs.jhu.edu/~alanlab/Pubs23/qu2023abdomenatlas.pdf) | [code](https://github.com/MrGiovanni/AbdomenAtlas) | [dataset](https://github.com/ljwztc/CLIP-Driven-Universal-Model#dataset) | [annotation](https://www.dropbox.com/scl/fi/28l5vpxrn212r2ejk32xv/AbdomenAtlas.tar.gz?rlkey=vgqmao4tgv51hv5ew24xx4xpm&dl=0) | [poster](document/neurips_poster.pdf)

<b>AbdomenAtlas-8K: Human-in-the-Loop Annotating Eight Anatomical Structures for 8,448 Three-Dimensional Computed Tomography Volumes in Three Weeks</b> <br/>
[Chongyu Qu](https://github.com/Chongyu1117)<sup>1</sup>, [Tiezheng Zhang](https://github.com/ollie-ztz)<sup>1</sup>, [Hualin Qiao](https://www.linkedin.com/in/hualin-qiao-a29438210/)<sup>2</sup>, [Jie Liu](https://ljwztc.github.io/)<sup>3</sup>, [Yucheng Tang](https://scholar.google.com/citations?hl=en&user=0xheliUAAAAJ)<sup>4</sup>, [Alan L. Yuille](https://www.cs.jhu.edu/~ayuille/)<sup>1</sup>, and [Zongwei Zhou](https://www.zongweiz.com/)<sup>1,*</sup> <br/>
<sup>1 </sup>Johns Hopkins University,  <br/>
<sup>2 </sup>Rutgers University,  <br/>
<sup>3 </sup>City University of Hong Kong,   <br/>
<sup>4 </sup>NVIDIA <br/>
RSNA 2023 (Oral Presentation) <br/>
[paper](document/rsna_abstract.pdf) | [code](https://github.com/MrGiovanni/AbdomenAtlas) | [slides](document/rsna_slides.pdf)

**&#9733; An improved version, AbdomenAtlas 1.1, can be found at [SuPreM](https://github.com/MrGiovanni/SuPreM)**

**&#9733; We have maintained a document for [Frequently Asked Questions](document/frequently_asked_questions.md).**

## 0. Installation

```bash
git clone https://github.com/MrGiovanni/AbdomenAtlas
```

See [installation instructions](document/INSTALL.md) to create an environment and obtain requirements.

## 1. Download AI models

We offer pre-trained checkpoints of Swin UNETR and U-Net. The models were trained on a combination of 14 publicly available CT datasets, consisting of 3,410 (see details in [CLIP-Driven Universal Model](https://github.com/ljwztc/CLIP-Driven-Universal-Model)).
Download the trained models and save them into `./pretrained_checkpoints/`.

| Architecture | Param | Download |
|  ----  | ----  |  ----  |
| U-Net  | 19.08M | [link](https://www.dropbox.com/s/lyunaue0wwhmv5w/unet.pth) |
| Swin UNETR | 62.19M | [link](https://www.dropbox.com/s/jdsodw2vemsy8sz/swinunetr.pth) |

## 2. Prepare your datasets

It can be publicly available datasets (e.g., BTCV) or your private datasets. Currently, we only take data formatted in `nii.gz`. This repository will help you assign annotations to these datasets, including 25 organs and six types of tumors (*where the annotation of eight organs is pretty accurate*).

##### 2.1 Download

Taking the BTCV dataset as an example, download this dataset and save it to the `datapath` directory.
```bash
cd $datapath
wget https://www.dropbox.com/s/jnv74utwh99ikus/01_Multi-Atlas_Labeling.tar.gz
tar -xzvf 01_Multi-Atlas_Labeling.tar.gz
```

##### 2.2 Preprocessing

Generate a list for this dataset.

```bash
cd AbdomenAtlas/
python -W ignore generate_datalist.py --data_path $datapath --dataset_name $dataname --folder img --out ./dataset/dataset_list --save_file $dataname.txt
```

## 3. Generate masks

##### U-Net
```bash
CUDA_VISIBLE_DEVICES=0 python -W ignore test.py --resume pretrained_checkpoints/unet.pth --backbone unet --save_dir $savepath --dataset_list $dataname --data_root_path $datapath --store_result >> logs/$dataname.unet.txt
```

##### Swin UNETR
```bash
CUDA_VISIBLE_DEVICES=0 python -W ignore test.py --resume pretrained_checkpoints/swinunetr.pth --backbone swinunetr --save_dir $savepath --dataset_list $dataname --data_root_path $datapath --store_result >> logs/$dataname.swinunetr.txt
```
To generate attention maps for the active learning process (Step 5 [optional]), remember to save entropy and soft predictions by using the options `--store_entropy` and `--store_soft_pred`

## 4. Data Assembly

In the assembly process, our utmost priority is given to the original annotations supplied by each public dataset. Subsequently, we assign secondary priority to the revised labels from our annotators. The pseudo labels, generated by AI models, are accorded the lowest priority. The following code can implement this priority into the assembled dataset.

```bash
python -W ignore assemble.py --data_path $savepath --dataset_name $dataname --backbone swinunetr --save_dir SAVE_DIR --version V1
```

This is how our AbdonmenAtlas-8K appears
```
    $savepath/
    ├── $dataname_img0001
    ├── $dataname_img0002
    ├── $dataname_img0003
        │── ct.nii.gz
        ├── original_label.nii.gz
        ├── pseudo_label.nii.gz
        └── segmentations
            ├── spleen.nii.gz
            ├── liver.nii.gz
            ├── pancreas.nii.gz
```

## 5. [Optional] Active Learning

If you want to perform the active learning process, you will need the following [active learning instructions](document/active_learning_instructions.md) to generate the attention map for human annotators.

<p align="center"><img width="100%" src="document/fig_attention_map.jpg" /></p>
Figure. Illustration of an attention map.

## TODO

- [x] Release pre-trained AI model checkpoints (U-Net and Swin UNETR)
- [x] Release the AbdomenAtlas-8K dataset (we commit to releasing 3,410 of the 8,448 CT volumes)
- [ ] Support more data formats (e.g., dicom)

## Citation 

```
@article{qu2023annotating,
  title={Annotating 8,000 Abdominal CT Volumes for Multi-Organ Segmentation in Three Weeks},
  author={Qu, Chongyu and Zhang, Tiezheng and Qiao, Hualin and Liu, Jie and Tang, Yucheng and Yuille, Alan and Zhou, Zongwei},
  journal={arXiv preprint arXiv:2305.09666},
  year={2023}
}
```

## Acknowledgements
This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research and partially by the Patrick J. McGovern Foundation Award. We appreciate the effort of the MONAI Team to provide open-source code for the community.
