import torch
import numpy as np
import random
import torch.nn.functional as F
import os

        #pseudo_label = pseudo_label.view(pred_remain1_W.shape[0], -1)
class PrototypeMemory:
    def __init__(self, feature_size=256, n_classes=21, gamma=0.99, resume1=None,resume2=None):
        self.feature_size = feature_size
        self.n_classes = n_classes
        #self.prototype = torch.zeros([n_classes, self.feature_size]).cuda()
        self.init(feature_size, n_classes, resume1=resume1,resume2=resume2)
        self.gamma = gamma

    # def update_prototype(self, feats, lbls, i_iter):
    #     if i_iter < 20:
    #         momentum = 0
    #     else:
    #         momentum = self.gamma
    #     for i_cls in torch.unique(lbls):
    #         if i_cls==255:
    #             continue
    #         feats_i = feats[lbls == i_cls, :]
    #         feats_i_center = feats_i.mean(dim=0, keepdim=True)
    #         self.prototype[i_cls, :] = self.prototype[i_cls, :] * \
    #                                 momentum + feats_i_center * (1 - momentum)
    #         self.prototype[i_cls, :] = feats_i_center
        # if norm:
        #     self.prototype = F.normalize(self.prototype)
            #self.batch_pro = F.normalize(self.batch_pro)
    def update_prototype(self, reps, labels):
        mask = (labels != 255)
        # remove IGNORE_LABEL pixels
        labels = labels[mask]
        reps = reps[mask]

        momentum = self.gamma
        #reps = reps.detach().cpu().numpy()
        reps = reps.detach()
        #labels = torch.from_numpy(labels).detach()
        labels = labels.detach()
        ids_unique = labels.unique()
        for i in ids_unique:
            i = i.item()
            mask_i = (labels == i)
            feature = reps[mask_i]
            feature = torch.mean(feature, dim=0)
            self.Proto[i, :] = (1 - momentum) * feature + self.Proto[i, :] * momentum

    def init(self, feature_num, n_classes=19, resume1="", resume2=""):
        if resume1:
            resume = os.path.join(resume1, 'prototype_feat_dist1.pth')
            print("Loading checkpoint from {}".format(resume))
            checkpoint = torch.load(resume, map_location=torch.device('cpu'))
            self.Proto = checkpoint['Proto'].cuda()
            # self.Amount = checkpoint['Amount'].cuda(non_blocking=True)
        elif resume2:
            resume = os.path.join(resume2, 'prototype_feat_dist2.pth')
            print("Loading checkpoint from {}".format(resume))
            checkpoint = torch.load(resume, map_location=torch.device('cpu'))
            self.Proto = checkpoint['Proto'].cuda()
            # self.Amount = checkpoint['Amount'].cuda(non_blocking=True)
        else:
            self.Proto = torch.zeros([n_classes, feature_num]).cuda()
            # self.Amount = torch.zeros(n_classes).cuda(non_blocking=True)


    def get_prototype_all_torch(self):
        # return torch.from_numpy(self.prototype)
        return self.Proto

    def save(self, name, output):
        torch.save({'Proto': self.Proto.cpu()
                    },
                   os.path.join(output, name))


