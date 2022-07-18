[environment]
pytorch=1.7
python=3.8.10
tqdm=4.54.1
pynvml=8.0.4

[dataset] 
Due to the oversize dataset（ beyond maximum limitation 100M, please download the public NTU RGB+D 120 dataset or use our subset when the work is accepted.

[run]
#run train.py with default parameters(DAST w/ RankMax, 5-way-1-shot, STGCN, on NTU RGB+D 120). 
python train.py --SA 0 --reg 0.1

#run train.py with spatial activateion(DAST w/ SA, 5-way-1-shot, STGCN, on NTU RGB+D 120). 
python train.py --SA 1 --reg 0

#run train.py with full model(DAST (full), 5-way-1-shot, STGCN, on NTU RGB+D 120). 
python train.py --SA 1 --reg 0.1