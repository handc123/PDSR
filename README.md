# PDSR
Code used for the results in the paper [A Pseudo-Dual Self-Rectification Framework for Semantic Segmentation](https://ieeexplore.ieee.org/abstract/document/10219865/)
Abstract.Semantic segmentation has achieved remarkable
success in various applications. However, the training process
for such techniques necessitates a significant amount of labeled
data. Although semi-supervised frameworks can alleviate this
issue, traditional approaches typically require multiple baseline
models to form a dual model. To allow a semi-supervised semantic
segmentation framework to be used in robotic systems with
precious computation and memory resources, we propose a
framework utilizing a single baseline model only. The overall
framework is composed of three parts: an encoder, a shallow
decoder, and a deep decoder. It distills knowledge from the
ensemble of two decoders to improve the encoder, which can
implicitly form a pseudo-dual model. It also calculates classwise likelihoods according to the similarity between features and
class prototypes learned from different decoders and rectifies
low-confidence pseudo-labels. Our framework outperforms stateof-the-art frameworks on benchmark datasets with a significant
amount of decrease in using computing resources.
## Datasets
mkdir ../data/CityScapes/  
Download the dataset from [here](https://www.cityscapes-dataset.com/).  
mkdir ../data/voc/  
Download the dataset from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).
## Experiments
python train_kl.py --snapshot-dir ./result/2/kl --drop 0.1 --batch-size 2 --learning-rate 1e-4 --crop-size 256,512  
python train_pro.py -snapshot-dir ./result/2/pro --drop 0.1 --batch-size 2 --learning-rate 1e-4 --crop-size 256,512  
## Citation
@inproceedings{hao2023pseudo,
  title={A Pseudo-Dual Self-Rectification Framework for Semantic Segmentation},
  author={Hao, Huazheng and Xiao, Hui and Dong, Li and Yan, Diqun and Liang, Dongtai and Zhuang, Jiayan and Peng, Chengbin},
  booktitle={2023 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={408--413},
  year={2023},
  organization={IEEE}
}
