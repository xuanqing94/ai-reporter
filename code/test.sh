#!/bin/bash


for level in 2.4 4.7 9.8 20.4 30.2 40.8 60.4 80.8 100; do
    result_d=./results/excitation_${level}
    mkdir $result_d | true
    rm $result_d/*.png | true
    python test.py --dataroot=dataset/excitation_${level}/AB \
        --name=excitation_${level} --gpu_ids=7 --model=pix2pix --input_nc=1 \
        --output_nc=1 --dataset_mode aligned --load_size=1024 --crop_size 1024 \
        --results_dir=./results/excitation_${level}
    #mv ./results/*.png $result_d
done
