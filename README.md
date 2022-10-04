# DASTM
The official implementation of ECCV22 paper "Learning Spatial-Preserved Skeleton Representations for Few-Shot Action Recognition"

# Environment
pytorch=1.7

python=3.8.10

tqdm=4.54.1

pynvml=8.0.4

# Datasets
The used datasets or their subsets can be found [here](https://drive.google.com/drive/folders/1lowLKFjUvVQPsnJzSQTGMSNZeaWy5JnP).

# Run

## Run train.py with full model(DASTM (full), 5-way-1-shot, STGCN, on NTU RGB+D 120). 
    python train.py --SA 1 --reg 0.1

##  Run train.py with default parameters(DASTM w/ RankMax, 5-way-1-shot, STGCN, on NTU RGB+D 120). 
    python train.py --SA 0 --reg 0.1

## Run train.py with spatial activation(DASTM w/ SA, 5-way-1-shot, STGCN, on NTU RGB+D 120). 
    python train.py --SA 1 --reg 0

## The other detailed usages are coming soon. Tue to the limited time, the code base is still being refined.

If you find this code is useful, please cite our paper in your work. Thanks!

@InProceedings{DASTM ,
author="Ning, Ma
and Hongyi, Zhang
and Xuhui, Li
and Sheng, Zhou
and Zhen, Zhang
and Jun, Wen"
and Jingjun, Gu"
and Haifeng, Li"
and Jiajun, Bu",
title="Learning Spatial-Preserved Skeleton Representations for Few-Shot Action Recognition",
booktitle="Computer Vision -- ECCV 2022",
year="2022",
publisher="Springer International Publishing",
}

