# DASTM
The official implementation of ECCV22 paper "Learning Spatial-Preserved Skeleton Representations for Few-Shot Action Recognition"

# Environment
pytorch=1.7
python=3.8.10
tqdm=4.54.1
pynvml=8.0.4

# Datasets
Due to the oversize dataset（ beyond maximum limitation 100M, please download the public NTU RGB+D 120 dataset or use our subset ([here](https://drive.google.com/drive/folders/1lowLKFjUvVQPsnJzSQTGMSNZeaWy5JnP)).

# Run

## Run train.py with full model(DASTM (full), 5-way-1-shot, STGCN, on NTU RGB+D 120). 
    python train.py --SA 1 --reg 0.1

## Run train.py with default parameters(DASTM w/ RankMax, 5-way-1-shot, STGCN, on NTU RGB+D 120). 
    python train.py --SA 0 --reg 0.1

## Run train.py with spatial activation(DASTM w/ SA, 5-way-1-shot, STGCN, on NTU RGB+D 120). 
    python train.py --SA 1 --reg 0

