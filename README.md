# PDSR
Code used for the results in the paper https://ieeexplore.ieee.org/abstract/document/10219865/
code
Download the data (VOC, Cityscapes) and pre-trained ResNet models from OneDrive link:
https://www.cityscapes-dataset.com/
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
Examples
Training a model with semi-supervised learning with example config on a single gpu
python train_kl.py --snapshot-dir ./result/2/kl --drop 0.1 --batch-size 2 --learning-rate 1e-4 --crop-size 256,512
