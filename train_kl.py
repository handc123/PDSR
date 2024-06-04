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
# from tensorboardX import SummaryWriter
import math

from data.voc_dataset import VOCDataSet,VOCGTDataSet
from data.cityscapes_dataset import cityscapesDataSet
from PrototypeMemory import PrototypeMemory
from model.deeplab_multi import DeeplabMulti
from multiprocessing import Pool
from utils.metric import ConfusionMatrix
from tqdm import tqdm
from utils.tool import adjust_learning_rate, adjust_learning_rate_D, Timer
from torch.utils.data.sampler import SubsetRandomSampler
#IMG_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
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
CROP_SIZE = '640, 360'
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
MAX_VALUE = 2
NUM_CLASSES = 21
NUM_STEPS = 100000
NUM_STEPS_STOP = 100000  # early stopping
POWER = 0.9
RANDOM_SEED = 0

#RESTORE_FROM = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
RESTORE_FROM = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'
#RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
#RESTORE_FROM = 'E:\Seg\\test\city\8_new\\best_model.pth'
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
TARGET = 'pascal_voc'
SET = 'train'
NORM_STYLE = 'bn' # or in
DATASET='pascal_voc'
#DATASET='pascal_voc'
SAVE_PATH = './test/8_voc/kl_11'
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
def  online_meaniou(pred, y, iou_t):
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

def load_state(path, model, optimizer=None, key="model_state_dict"):
    #rank = dist.get_rank()

    def map_func(storage, location):
        return storage.cuda(1)

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



def main():
    """Create the model and start the training."""


    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    w, h = map(int, args.input_size.split(','))
    args.input_size = (w, h)

    w, h = map(int, args.crop_size.split(','))
    args.crop_size = (w, h)
    model = DeeplabMulti(num_classes=args.num_classes, use_se=args.use_se, train_bn=args.train_bn,
                         norm_style=args.norm_style, droprate=args.droprate)

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
    '''
    在PyTorch中，可以使用torchstat这个库来查看网络模型的一些信息，包括总的参数量params、MAdd、显卡内存占用量和FLOPs等
    pip install torchstat
    '''


    model = model.cuda(1)
    gen_opt = optim.SGD(model.optim_parameters(args),
                        lr=args.learning_rate, momentum=args.momentum,nesterov=True,weight_decay=args.weight_decay)
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
                                   scale=False, mirror=False, mean=IMG_MEAN, augment=False, flip=False)
        train_dataset_remain = VOCDataSet(args.data_dir, args.uldata_list,
                                   crop_size=args.crop_size,
                                   scale=False, mirror=False, mean=IMG_MEAN, augment=False, flip=False)

    elif args.dataset == 'cityscapes':

        train_dataset = cityscapesDataSet(args.data_dir, args.ldata_list,
                                                     crop_size=args.crop_size,set=args.set,
                                                     scale=True, mirror=False, mean=IMG_MEAN, augment=False, flip=False)

        train_dataset_remain = cityscapesDataSet(args.data_dir, args.uldata_list,
                                                 crop_size=args.crop_size,set=args.set,
                                                 scale=True, mirror=False, mean=IMG_MEAN, augment=True, flip=False)



    trainloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True)
    trainloader_remain = data.DataLoader(train_dataset_remain, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers,
                                         pin_memory=True, drop_last=True)

    trainloader_iter = iter(trainloader)
    trainloader_remain_iter = iter(trainloader_remain)
    if args.tensorboard:
        args.log_dir += '/'+ os.path.basename(args.snapshot_dir)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        #writer = SummaryWriter(args.log_dir)

    if args.fp16:
        # Name the FP16_Optimizer instance to replace the existing optimizer
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."
        model, gen_opt = amp.initialize(model, gen_opt, opt_level="O1")
    interp = nn.Upsample(size=args.crop_size, mode='bilinear', align_corners=True)
    interp_ = nn.Upsample(size=args.input_size, mode='bilinear', align_corners=True)
    seg_loss = nn.CrossEntropyLoss(ignore_index=255)

    sm = torch.nn.Softmax(dim=1)
    kl_loss = nn.KLDivLoss(size_average=False)
    log_sm = torch.nn.LogSoftmax(dim=1)
    best_mIoU = 0

    filename = os.path.join(SAVE_PATH, 'result.txt')

    for i_iter in range(last_iter,args.num_steps):

        loss_seg_value1 = 0
        loss_seg_value2 = 0
        adjust_learning_rate(gen_opt, i_iter, args)
        for sub_i in range(args.iter_size):
            try:
                batch = next(trainloader_iter)
            except:
                trainloader_iter=iter(trainloader)
                batch = next(trainloader_iter)
            try:
                batch_remain = next(trainloader_remain_iter)
            except:
               trainloader_remain_iter = iter(trainloader_remain)
               batch_remain = next(trainloader_remain_iter)
            images, labels, _, _, = batch
            images = images.cuda(1)
            labels = labels.long().cuda(1)
            b, H, W = labels.size()
            images_remain_W,  labels_remain, _, _ = batch_remain
            images_remain_W = images_remain_W.cuda(1)
            #images_remain_S = images_remain_S.cuda()

            with Timer("Elapsed time in update: %f"):
                gen_opt.zero_grad()
                pred1, pred2,features1, features2= model(images)
                pred1 = interp(pred1)
                pred2 = interp(pred2)

                loss_seg1 = seg_loss(pred1, labels)
                loss_seg2 = seg_loss(pred2, labels)
                loss = loss_seg2 + 0.4*loss_seg1  # 有标签损失
                pred_remain1_W, pred_remain2_W,features1_W, features2_W = model(images_remain_W)
                pred_remain1_W = interp(pred_remain1_W)
                pred_remain2_W = interp(pred_remain2_W)
                n, c, h, w = pred_remain1_W.shape
                loss_kl2 = 0.0
                if i_iter < 0:
                  lambda_kl_target_copy = 0

                else:
                   lambda_kl_target_copy = args.lambda_kl_target



                if lambda_kl_target_copy > 0:

                    with torch.no_grad():

                        mean_pred = sm(
                            pred_remain1_W +  pred_remain2_W)
                    loss_kl2 = kl_loss(log_sm(pred_remain1_W), mean_pred)/ (n * h * w) + kl_loss(
                        log_sm(pred_remain2_W),
                        mean_pred) / (n * h * w)

                    loss = loss + 0.1*loss_kl2




                loss.backward()
                gen_opt.step()

                loss_seg_value1 += loss_seg1.item() / args.iter_size
                loss_seg_value2 += loss_seg2.item() / args.iter_size

            del pred1, pred2, pred_remain1_W, pred_remain2_W

            if args.tensorboard:
                scalar_info = {
                    'loss_seg1': loss_seg_value1,
                    'loss_seg2': loss_seg_value2,


                    'loss_kl_target': loss_kl2,
                    #'val_loss': val_loss,
                }

                # if i_iter % 100 == 0:
                #     for key, val in scalar_info.items():
                #         writer.add_scalar(key, val, i_iter)

            print('exp = {}'.format(args.snapshot_dir))
            print(
               '\033[1m iter = %8d/%8d \033[0m loss_seg1 = %.4f loss_seg2 = %.4f  loss_kl2 = %.4f ' % (
               i_iter, args.num_steps, loss_seg_value1, loss_seg_value2,loss_kl2
               ))
            # print(
            #     '\033[1m iter = %8d/%8d \033[0m loss_seg1 = %.4f loss_seg2 = %.4f ' % (
            #         i_iter, args.num_steps, loss_seg_value1, loss_seg_value2,
            #     ))

            # clear loss
            del loss_seg1, loss_seg2, loss_kl2
            #del loss_seg1, loss_seg2

            if i_iter >= args.num_steps_stop - 1:
                print('save model ...')
                path = osp.join(args.snapshot_dir, 'VOC_' + str(args.num_steps_stop) + '.pth')
                check_point = {
                "epoch": i_iter,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": gen_opt.state_dict(),
                }

                torch.save(check_point,path
                           )
                # torch.save(Trainer.D1.state_dict(), osp.join(args.snapshot_dir, 'VOC_' + str(args.num_steps_stop) + '_D1.pth'))
                # torch.save(Trainer.D2.state_dict(), osp.join(args.snapshot_dir, 'VOC_' + str(args.num_steps_stop) + '_D2.pth'))
                break

            if i_iter % args.save_pred_every == 0 and i_iter != 0:
                print('taking snapshot ...')
                path = osp.join(args.snapshot_dir, 'last.pth')
                check_point = {
                    "epoch": i_iter,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": gen_opt.state_dict(),

                }

                torch.save(check_point, path
                           )
            if i_iter == 40000 :
                print('taking snapshot ...')
                path = osp.join(args.snapshot_dir, '20000.pth')
                check_point = {
                    "epoch": i_iter,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": gen_opt.state_dict(),

                }

                torch.save(check_point, path
                           )
            if i_iter % 1000 == 0 and i_iter != 0:
                data_list = []
                model.eval()
                # test_dataset = VOCGTDataSet(args.data_dir, args.val_data_list, crop_size=(321, 321),
                #                           scale=False, mirror=False, mean=IMG_MEAN)
                if args.dataset =='pascal_voc':

                    test_dataset = VOCDataSet(args.data_dir, args.val_data_list, args.input_size,
                                              scale=False, mirror=False, mean=IMG_MEAN, augment=False, flip=False)
                    #interp = nn.Upsample(size=(321, 321), mode='bilinear', align_corners=True)
                elif args.dataset == 'cityscapes':

                    test_dataset = cityscapesDataSet(args.data_dir, args.val_data_list, args.crop_size,
                                              scale=False, mirror=False, mean=IMG_MEAN, augment=False, flip=False)
                    #interp = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)

                testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=True)




                iou_t3 = torch.zeros(args.num_classes, args.num_classes).cuda(1)
                for index, batch in enumerate(testloader):
                    image, label, size, name = batch
                    size = size[0]
                    with torch.no_grad():
                        output1, output2,_,_ = model(Variable(image).cuda(1))
                        out3 = interp_(0.4*output1+output2)

                        label_cuda = Variable(label.long()).cuda(1)


                        iou_t3, iou3, mean_iou3 = online_meaniou(out3, label_cuda, iou_t3)

                #mIoU = get_iou(data_list, args.num_classes, args.dataset, filename)
                print(iou3)
                print(mean_iou3)



                if filename:
                    with open(filename, 'a') as f:
                        f.write('iter:' + str(i_iter) + '\n')
                        f.write('meanIOU: ' + str(iou3) + '\n')
                        f.write('meanIOU: ' + str(mean_iou3) + '\n')

                if mean_iou3 > best_mIoU:
                    best_mIoU = mean_iou3
                    path = osp.join(args.snapshot_dir,  f'best_model.pth')
                    # torch.save({'state_dict': model.state_dict(), 'epoch': i_iter, 'optimizer': gen_opt.state_dict()}, path)
                    torch.save(model.state_dict(), path)


        # if args.tensorboard:
        #     writer.close()
if __name__ == '__main__':
    main()