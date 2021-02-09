#!/bin/bash

python scripts/combine_A_and_B.py \
    --fold_A=dataset/neil_data/A \
    --fold_B=dataset/neil_data/B \
    --fold_AB=dataset/neil_data/AB
    
python train.py --dataroot=dataset/neil_data/AB \
    --name neil_f3 \
    --gpu_ids 1,2,5 --model=pix2pix --input_nc=1 --output_nc=1 \
    --dataset_mode aligned --batch_size 15 --load_size=1200 \
    --crop_size 1024
