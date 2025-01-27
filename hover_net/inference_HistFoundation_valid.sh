#!/bin/bash

for idx in $(seq 81 90); do 
# for idx in $(seq 47 91); do
    split=$(printf %03d $idx)

    # python run_infer.py --gpu 0 --nr_types 5 --type_info_path type_info.json --model_path weights/hovernet_original_kumar_notype_tf2pytorch.tar --model_mode original --batch_size 16 --nr_inference_workers 16 --nr_post_proc_workers 16  tile --input_dir /data/SR-Hist-Foundation/HR_valid/split_$split --output_dir /data/SR-Hist-Foundation/HR_valid_hover/split_$split
    python run_infer.py --gpu 1 --nr_types 5 --type_info_path type_info.json --model_path weights/hovernet_original_kumar_notype_tf2pytorch.tar --model_mode original --batch_size 16 --nr_inference_workers 16 --nr_post_proc_workers 16  tile --input_dir /data/SR-Hist-Foundation/LR-x4_valid_up/split_$split --output_dir /data/SR-Hist-Foundation/LR-x4_valid_up_hover/split_$split

done