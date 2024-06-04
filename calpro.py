import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
import time
import yaml
from tensorboardX import SummaryWriter
import math
from collections import Counter
from model.deeplabv2 import Deeplab
from utils.tool import adjust_learning_rate, adjust_learning_rate_D, Timer


import utils.imutils as imutils
from torchvision import transforms
from data.voc_dataset import VOCDataSet,VOCGTDataSet
from data.cityscapes_dataset import cityscapesDataSet
from PrototypeMemory import PrototypeMemory
from model.deeplab_multi import DeeplabMulti
from multiprocessing import Pool
from utils.metric import ConfusionMatrix
from tqdm import tqdm

from torch.utils.data.sampler import SubsetRandomSampler
#IMG_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
AUTOAUG = False
AUTOAUG_TARGET = False
MODEL = 'DeepLab'
BATCH_SIZE = 8
ITER_SIZE = 1
NUM_WORKERS = 0
DATA_DIRECTORY = './data/voc_dataset/'
DATA_LIST_PATH = './data/voc_list/train_aug.txt'
Label_DATA_LIST_PATH = './data/voc_list/train_aug_labeled_1-4.txt'
Unlabel_DATA_LIST_PATH = './data/voc_list/train_aug_unlabeled_1-4.txt'
Val_DATA_LIST_PATH = './data/voc_list/val.txt'
# DATA_DIRECTORY = './data/Cityscapes/data'
# DATA_LIST_PATH = './data/cityscapes_list/train.txt'
# Label_DATA_LIST_PATH = './data/cityscapes_list/train_aug_labeled_1-8.txt'
# Unlabel_DATA_LIST_PATH = './data/cityscapes_list/train_aug_unlabeled_1-8.txt'
# Val_DATA_LIST_PATH = './data/cityscapes_list/val.txt'
DROPRATE = 0.1
IGNORE_LABEL = 255
INPUT_SIZE = '505,505'
#DATA_DIRECTORY_TARGET = './data/Cityscapes/data'
#DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
CROP_SIZE = '321, 321'
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
MAX_VALUE = 2
NUM_CLASSES = 21
NUM_STEPS = 100000
NUM_STEPS_STOP = 100000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
#RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
RESTORE_FROM = './test/16_voc/sup/best_model.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
WARM_UP = 0 # no warmup
LOG_DIR = './log'
SPLIT_ID = None

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_KL_TARGET = 0
LABELED_RATIO=0.125
TARGET = 'cityscapes'
SET = 'train'
NORM_STYLE = 'bn' # or in
#DATASET='cityscapes'
DATASET='pascal_voc'
SAVE_PATH = './test/8_voc/sup'
split_id='./splits/voc/split_2.pkl'
try:
    import apex
    from apex import amp
    from apex.fp16_utils import *
except ImportError:
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--autoaug", action='store_true', help="use augmentation or not" )
    parser.add_argument("--autoaug_target", action='store_true', help="use augmentation or not" )
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    #parser.add_argument("--target", type=str, default=TARGET,
                       # help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")

    parser.add_argument("--data_list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ldata_list", type=str, default=Label_DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--uldata_list", type=str, default=Unlabel_DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--val_data_list", type=str, default=Val_DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--droprate", type=float, default=DROPRATE,
                        help="DropRate.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--crop-size", type=str, default=CROP_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--split-id", type=str, default=SPLIT_ID,
                        help="split order id")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-kl-target", type=float, default=LAMBDA_KL_TARGET,
                       help="lambda_me for minimize kl loss on target.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--max-value", type=float, default=MAX_VALUE,
                        help="Max Value of Class Weight.")
    parser.add_argument("--norm-style", type=str, default=NORM_STYLE,
                        help="Norm Style in the final classifier.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--labeled-ratio", type=float, default=LABELED_RATIO,
                        help="ratio of the labeled data to full dataset")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--warm-up", type=float, default=WARM_UP, help = 'warm up iteration')
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--class-balance", action='store_true', help="class balance.")
    parser.add_argument("--use-se", action='store_true', help="use se block.")
    parser.add_argument("--only-hard-label",type=float, default=0,
                         help="class balance.")
    parser.add_argument("--train_bn", action='store_true', help="train batch normalization.")
    parser.add_argument("--sync_bn", action='store_true', help="sync batch normalization.")
    parser.add_argument("--often-balance", action='store_true', help="balance the apperance times.")
    parser.add_argument("--gpu-ids", type=str, default='0', help = 'choose gpus')
    parser.add_argument("--tensorboard", action='store_false', help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="dataset name pascal_voc or pascal_context")
    parser.add_argument('--resume',action='store_false', help="use resume.")
    #parser.add_argument('--local_rank', help='experiment configuration filename', default="lib/config/360CC_config.yaml", type=int)
    return parser.parse_args()


args = get_arguments()
if not os.path.exists(args.snapshot_dir):
    os.makedirs(args.snapshot_dir)

with open('%s/opts.yaml'%args.snapshot_dir, 'w') as fp:
    yaml.dump(vars(args), fp, default_flow_style=False)
def online_meaniou(pred, y, iou_t):
    pred_argmax = torch.argmax(pred,dim=1)
    for i in range(pred.size(1)):
        for j in range(pred.size(1)):
            iou_t[i][j] += ((pred_argmax == j) & (y == i)).sum()

    iou = []
    for i in range(pred.size(1)):
        if ((iou_t[i, :].sum() + iou_t[:, i].sum()) - iou_t[i][i]) > 0:
            iou.append(iou_t[i][i] / ((iou_t[i, :].sum() + iou_t[:, i].sum()) - iou_t[i][i]))
        else:
            iou.append(0.)

    mean_iou = sum(iou) / len(iou)

    return iou_t, iou, mean_iou
def train_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()
def get_iou(args, data_list, class_num, save_path=None):

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()

    if args.dataset == 'pascal_voc':
        classes = np.array(('background',  # always index 0
                            'aeroplane', 'bicycle', 'bird', 'boat',
                            'bottle', 'bus', 'car', 'cat', 'chair',
                            'cow', 'diningtable', 'dog', 'horse',
                            'motorbike', 'person', 'pottedplant',
                            'sheep', 'sofa', 'train', 'tvmonitor'))
    elif args.dataset == 'cityscapes':
        classes = np.array(("road", "sidewalk",
                            "building", "wall", "fence", "pole",
                            "traffic_light", "traffic_sign", "vegetation",
                            "terrain", "sky", "person", "rider",
                            "car", "truck", "bus",
                            "train", "motorcycle", "bicycle"))

    for i, iou in enumerate(j_list):
        if j_list[i] > 0:
            print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i]))

    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            for i, iou in enumerate(j_list):
                f.write('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i]) + '\n')
            f.write('meanIOU: ' + str(aveJ) + '\n')

def label_distribution(dataloader,args):
    class_count=[0 for i in range(args.num_classes)]

    for images,labels,_,_,_ in tqdm(dataloader):
        labels = torch.tensor(labels)
        class_unique=torch.unique(labels)
        for i in class_unique:
            if i==args.ignore_label:
                continue
            class_count[i]+=torch.sum(labels==i).item()
    print(class_count)
    sampling=[]
    res = [0 for i in range(args.num_classes)]
    sorted_id = sorted(range(len(class_count)), key=lambda k: class_count[k], reverse=True)
    class_count_sort=sorted(class_count,reverse=True)
    print(class_count_sort)
    for i in range(len(class_count_sort)):
        sampling.append(math.pow(class_count_sort[len(class_count_sort) - i - 1] / class_count_sort[0], 1 / 4))
    for i in range(len(class_count)):
        res[sorted_id[i]]=sampling[i]
    return res


def generate_class_pseudo_labels_every_iter(pseudo_probability, pseudo_label,is_print,class_sampling=[1 for i in range(args.num_classes)]):
    probabilities = [np.array([], dtype=np.float32) for _ in range(args.num_classes)]
    for j in range(args.num_classes):
        probabilities[j] = np.concatenate((probabilities[j], pseudo_probability[pseudo_label == j].detach().cpu().numpy()))

    # Sort (n * log(n) << n * label_ratio, so just sort is good) and find kc
    kc = []
    # exceptions=[]
    # for j in range(args.num_classes):
    #     if len(probabilities[j]) == 0:
    #         with open('exceptions.txt', 'a') as f:
    #             f.write(str(time.asctime()) + '--' + str(j) + '\n')


    #长尾标签采样，尾部标签的精确率高，如果预测为尾标签，大概率是正确的

    for j in range(args.num_classes):
        probabilities[j].sort()
        probabilities[j]=probabilities[j][::-1]
        if len(probabilities[j])==0:
            kc.append(0.00001)
        else:
            kc.append(probabilities[j][int(len(probabilities[j])*class_sampling[j]) - 1])
    del probabilities  # Better be safe than...
    if is_print==1:
        print(kc)

    return kc
def generate_pseudo_label(values, pseudo_label, class_sampling):
        cbst_thresholds = generate_class_pseudo_labels_every_iter(values, pseudo_label, 0, class_sampling)
        for j in range(args.num_classes):
            pseudo_label[((pseudo_label == j) * (values < cbst_thresholds[j]))] = 255
        return pseudo_label


def generate_mask_label(num_classes,label):
        mask_temp = list()
        mask_class = list()

        for i in range(num_classes):
            mask_class.append((label == i).to(dtype=torch.float32))

        for i in label:
            if i==255:
                mask_temp.append(torch.zeros_like(label))
                #mask_temp.append(mask_class[i])
            else :
                mask_temp.append(mask_class[i])
                #mask_temp.append(torch.zeros_like(label))
        mask = torch.stack(mask_temp)

        return mask.cuda()
def load_state(path, model, optimizer=None, key="model_state_dict"):
    #rank = dist.get_rank()

    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):

        print("=> loading checkpoint '{}'".format(path))

        checkpoint = torch.load(path, map_location=map_func)

        # fix size mismatch error
        ignore_keys = []
        state_dict = checkpoint[key]

        for k, v in state_dict.items():
            if k in model.state_dict().keys():
                v_dst = model.state_dict()[k]
                if v.shape != v_dst.shape:
                    ignore_keys.append(k)

                    print(
                            "caution: size-mismatch key: {} size: {} -> {}".format(
                                k, v.shape, v_dst.shape
                            )
                        )

        for k in ignore_keys:
            checkpoint.pop(k)

        model.load_state_dict(state_dict, strict=False)


        ckpt_keys = set(state_dict.keys())
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        for k in missing_keys:
                print("caution: missing keys from checkpoint {}: {}".format(path, k))

        if optimizer is not None:

            last_iter = checkpoint["epoch"]
            proto_w = checkpoint["pro_w"]
            proto_s = checkpoint["pro_s"]
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            print(
                    "=> also loaded optimizer from checkpoint '{}' (epoch {})".format(
                        path, last_iter
                    )
                )
            return last_iter,proto_w,proto_s
    else:

        print("=> no checkpoint found at '{}'".format(path))
def GCN_feature(features,label_u,b,model):
    new_features = list()
    for i in range(0, b):
        rep_batch_temp = features[i]
        norm = torch.norm(rep_batch_temp, 2, 1).view(-1, 1)
        adj = torch.div(torch.mm(rep_batch_temp, rep_batch_temp.t()), torch.mm(norm, norm.t()) + 1e-7)
        adj = torch.softmax(adj, dim=1)
        mask = generate_mask_label(args.num_classes, label_u[i])
        adj = adj * mask
        adj = (adj.t() / (adj.sum(1) + 1e-7)).t()
        GCN_feature = model.module.forward_GCN(rep_batch_temp, adj)
        new_features.append(GCN_feature)
    new_features = torch.stack(new_features)
    return new_features
def main():
    """Create the model and start the training."""
    args = get_arguments()

    config_path = os.path.join(os.path.dirname(args.restore_from), 'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    w, h = map(int, config['crop_size'].split(','))
    args.crop_size = (w, h)
    args.model = config['model']
    print('ModelType:%s' % args.model)
    print('NormType:%s' % config['norm_style'])
    # batchsize = args.batchsize
    args.num_classes = config['num_classes']
    args.dataset = config['dataset']
    model_name = os.path.basename(os.path.dirname(args.restore_from))
    # args.save += model_name




    prototype_memory = PrototypeMemory(256, args.num_classes)
    prototype_memory_2 = PrototypeMemory(256, args.num_classes)

    model = Deeplab(num_classes=args.num_classes)

    last_iter = 0
    if args.restore_from[:4] == 'http':
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
    model.load_state_dict(new_params)
    # if args.restore_from[:4] == 'http':
    #     saved_state_dict = model_zoo.load_url(args.restore_from)
    # else:
    #     saved_state_dict = torch.load(args.restore_from)
    #
    #     # Copy loaded parameters to model
    # new_params = model.state_dict().copy()
    # for name, param in new_params.items():
    #     if name in saved_state_dict and param.size() == saved_state_dict[name].size():
    #         new_params[name].copy_(saved_state_dict[name])
    # model.load_state_dict(new_params)

    model = model.cuda(1)
    # if len(device_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    #     model = model.cuda()
    gen_opt = optim.SGD(model.optim_parameters(args),
                        lr=args.learning_rate, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
    args.resume = False
    if args.resume:
        #saved_state_dict = torch.load(args.restore_from)
        lastest_model = os.path.join('./test/city/8_new', "VOC_10000.pth")
        last_iter, prototype_memory_W,prototype_memory_S = load_state(
            lastest_model, model, optimizer=gen_opt, key="model_state_dict"
        )
    #model = torch.load(args.resume)

    if args.dataset =='pascal_voc':


        train_dataset = VOCDataSet(args.data_dir, args.ldata_list,
                                   crop_size=args.crop_size,
                                   scale=True, mirror=False, mean=IMG_MEAN, augment=False, flip=False)
        # train_dataset_size = len(train_dataset)
        # if split_id is not None:
        #     train_ids = pickle.load(open(split_id, 'rb'))
        #     print('loading train ids from {}'.format(split_id))
        # else:
        #     np.random.seed(RANDOM_SEED)
        #     train_ids = np.arange(train_dataset_size)
        #     np.random.shuffle(train_ids)
        #
        # train_sampler = data.sampler.SubsetRandomSampler(train_ids[:529])

        # train_dataset_remain = VOCDataSet(args.data_dir, args.uldata_list,
        #                                   crop_size=args.input_size,
        #                                   scale=True, mirror=False, mean=IMG_MEAN, augment=True, flip=True)
    elif args.dataset == 'cityscapes':

        train_dataset = cityscapesDataSet(args.data_dir, args.ldata_list,
                                                     crop_size=args.crop_size,set=args.set,
                                                     scale=True, mirror=False, mean=IMG_MEAN, augment=False, flip=False)

        # train_dataset_remain = cityscapesDataSet(args.data_dir, args.uldata_list,
        #                                          crop_size=args.crop_size,set=args.set,
        #                                          scale=True, mirror=False, mean=IMG_MEAN, augment=True, flip=True)



    trainloader = data.DataLoader(train_dataset, batch_size=1, shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True)


    trainloader_iter = iter(trainloader)

    if args.tensorboard:
        args.log_dir += '/'+ os.path.basename(args.snapshot_dir)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        writer = SummaryWriter(args.log_dir)

    if args.fp16:
        # Name the FP16_Optimizer instance to replace the existing optimizer
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."
        model, gen_opt = amp.initialize(model, gen_opt, opt_level="O1")
    interp = nn.Upsample(size=args.crop_size, mode='bilinear', align_corners=True)
    seg_loss = nn.CrossEntropyLoss(ignore_index=255)


    iteration = 0
    max_iters = len(trainloader)
    for batch in tqdm(trainloader):

        images, labels, _, _ = batch
        images = images.cuda(1)
        labels = labels.long().cuda(1)
        b, H, W = labels.size()

        gen_opt.zero_grad()
        pred, features = model(images)
        b, n, h_d, w_d = pred.size()

        features = features.contiguous().view(b * h_d * w_d, 256)

        label_down = nn.functional.interpolate(labels.clone().float().unsqueeze(1),
                                                      size=(h_d, w_d),
                                                      mode='nearest').squeeze(1).long()
        label_down = label_down.contiguous().view(b * h_d * w_d, )
        prototype_memory.update_prototype(features, label_down)

        iteration = iteration + 1
        if iteration == max_iters:
            prototype_memory.save(name='prototype_feat_dist.pth', output='./Res/cross')


if __name__ == '__main__':
    main()