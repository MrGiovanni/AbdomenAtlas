# Active Learning Instructions
##### Single Model

To generate the attention map and priority list with results from just one AI model, simply run the following command. In this scenario, the inconsistency will be 0.

```bash
python -W ignore create_attention.py --dataset_list $dataname --data_root_path $savepath --model_list swinunetr --priority --priority_name priority 
```

##### Multi-model

To generate the attention map and priority list using results from multiple AI models, begin by running the following command to obtain the average segmentation results from all the given models.

```bash
python -W ignore create_dataset.py --dataset_list $dataname  --data_root_path /$savepath --save_dir $savepath --model_list swinunetr unet nnunet --create_dataset --cpu
```
Next, run the following command to obtain the attention maps and priority list for the results from multiple models.

```bash
python -W ignore create_attention.py --dataset_list $dataname --data_root_path $savepath --model_list swinunetr --priority --priority_name priority 
```