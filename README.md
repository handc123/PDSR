## Code used for the results in the paper (A Pseudo-Dual Self-Rectification Framework for Semantic Segmentation)[https://ieeexplore.ieee.org/abstract/document/10219865/]

Download the data (VOC, Cityscapes) and pre-trained ResNet models from OneDrive link: (city)[https://www.cityscapes-dataset.com/]
(voc)[http://host.robots.ox.ac.uk/pascal/VOC/voc2012/]

# RUN
## example
python train_kl.py --snapshot-dir ./result/2/kl --drop 0.1 --batch-size 2 --learning-rate 1e-4 --crop-size 256,512
