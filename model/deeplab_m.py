import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.nn.parameter import Parameter
affine_par = True
import torch.nn.functional as F


def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def NormLayer(norm_dim, norm_style = 'bn'):
    if norm_style == 'bn':
        norm_layer = nn.BatchNorm2d(norm_dim)
    elif norm_style == 'in':
        norm_layer = nn.InstanceNorm2d(norm_dim, affine = True)
    elif norm_style == 'ln':
        norm_layer = nn.LayerNorm(norm_dim,  elementwise_affine=True)
    elif norm_style == 'gn':
        norm_layer = nn.GroupNorm(num_groups=32, num_channels=norm_dim, affine = True)
    return norm_layer

class SEBlock(nn.Module):
    def __init__(self, inplanes, r = 16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.se = nn.Sequential(
                nn.Linear(inplanes, inplanes//r),
                nn.ReLU(inplace=True),
                nn.Linear(inplanes//r, inplanes),
                nn.Sigmoid()
        )
    def forward(self, x):
        xx = self.global_pool(x)
        xx = xx.view(xx.size(0), xx.size(1))
        se_weight = self.se(xx).unsqueeze(-1).unsqueeze(-1)
        return x.mul(se_weight)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, train_bn = False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = train_bn

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = train_bn
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = train_bn
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes, norm_style = 'bn', droprate = 0.1, use_se = False):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        self.conv2d_list.append(
                nn.Sequential(*[
                nn.Conv2d(inplanes, 256, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                NormLayer(256, norm_style),
                nn.ReLU(inplace=True) ]))

        for dilation, padding in zip(dilation_series, padding_series):
            #self.conv2d_list.append(
            #    nn.BatchNorm2d(inplanes))
            self.conv2d_list.append(
                nn.Sequential(*[
                #nn.ReflectionPad2d(padding),
                nn.Conv2d(inplanes, 256, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True), 
                NormLayer(256, norm_style),
                nn.ReLU(inplace=True) ]))
 
        if use_se:
            self.bottleneck = nn.Sequential(*[SEBlock(256 * (len(dilation_series) + 1)),
                        nn.Conv2d(256 * (len(dilation_series) + 1), 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) ,
                        NormLayer(256, norm_style) ])
        else:
            self.bottleneck = nn.Sequential(*[
                nn.Conv2d(256 * (len(dilation_series) + 1), 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) ,
                NormLayer(256, norm_style) ])

        self.head = nn.Sequential(*[nn.Dropout2d(droprate),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=False) ])

        ##########init#######
        for m in self.conv2d_list:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.bottleneck:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.head:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)

    def forward(self, x,get_feat=False):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out = torch.cat( (out, self.conv2d_list[i+1](x)), 1)
        out = self.bottleneck(out)
        if get_feat:
            out_dict = {}
            out = self.head[0](out)
            out_dict['feat'] = out
            out = self.head[1](out)
            out_dict['out'] = out
            return out_dict
        else:
            out = self.head(out)
            return out
        #out = self.head(out)
        #return out



class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #support = torch.mm(input, self.weight)
        #output = torch.spmm(adj, support)
        support=input
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
class ResNetMulti(nn.Module):
    def __init__(self, block, layers, num_classes, use_se = False, train_bn = False, norm_style = 'bn', droprate = 0.1):
        self.inplanes = 64
        self.num_classes = num_classes
        self.train_bn = train_bn
        super(ResNetMulti, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = self.train_bn
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes, norm_style, droprate, use_se)
        self.layer6 = self._make_pred_layer(Classifier_Module, 2048 , [6, 12, 18, 24], [6, 12, 18, 24], num_classes, norm_style, droprate, use_se)


        '''
        self.projection_head1 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256))
        '''
        # self.projection_head2 = nn.Sequential(
        #     nn.Conv2d(1024+2048, 256,1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 1))
        '''
        self.projection_head1 = nn.Sequential(
            nn.Conv2d(1024, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1))
        '''
        # self.gc1 = GraphConvolution(256, 256)
        # self.gc2 = GraphConvolution(256, 256)
        # self.dropout = 0.3

        for m in self.modules():
           if isinstance(m, nn.Conv2d):
               n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
               m.weight.data.normal_(0, 0.01)
           elif isinstance(m, nn.BatchNorm2d):
               m.weight.data.fill_(1)
               m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = self.train_bn
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, train_bn = self.train_bn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, downsample = None, train_bn = self.train_bn))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes, norm_style, droprate, use_se):
        return block(inplanes, dilation_series, padding_series, num_classes, norm_style, droprate, use_se)


    def forward_projection_head(self, features):
        return self.projection_head1(features)

    def forward_projection_head2(self, features1):
        return self.projection_head2(features1)

    def forward_GCN(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

    def generate_class_pseudo_labels_every_iter(self,pseudo_probability, pseudo_label, is_print,class_sampling=[1 for i in range(19)]):
        #class_sampling=[1 for i in range(self.num_classes)]
        probabilities = [np.array([], dtype=np.float32) for _ in range(self.num_classes)]
        for j in range(self.num_classes):
            probabilities[j] = np.concatenate((probabilities[j], pseudo_probability[pseudo_label == j].cpu().numpy()))

        # Sort (n * log(n) << n * label_ratio, so just sort is good) and find kc
        kc = []
        # exceptions=[]
        # for j in range(args.num_classes):
        #     if len(probabilities[j]) == 0:
        #         with open('exceptions.txt', 'a') as f:
        #             f.write(str(time.asctime()) + '--' + str(j) + '\n')

        # 长尾标签采样，尾部标签的精确率高，如果预测为尾标签，大概率是正确的

        for j in range(self.num_classes):
            a = probabilities[j]
            a.sort()
            b=a
            probabilities[j].sort()
            probabilities[j] = probabilities[j][::-1]
            if len(probabilities[j]) == 0:
                kc.append(0.00001)
            else:
                kc.append(probabilities[j][int(len(probabilities[j]) * class_sampling[j]) - 1])
        del probabilities  # Better be safe than...
        if is_print == 1:
            print(kc)

        return kc
    # def generate_pseudo_label(self,values, pseudo_label,selected_label,classwise_acc):
    #     #selected_label = torch.tensor(selected_label)
    #     #selected_label = selected_label.cuda()
    #     #classwise_acc = torch.tensor(classwise_acc).cuda()
    #     select = values>0.95
    #     print(pseudo_label[select == 1] == 0)
    #     #select = values.ge(0.95).long()
    #     #select = values>0.95
    #     for i in range(self.num_classes):
    #         #selected_label[i] += torch.sum((pseudo_label[select == 1] == i))
    #         selected_label[i] += ((pseudo_label[select == 1] == i)).cpu().numpy().sum()
    #     classwise_acc = np.array(classwise_acc)
    #     thrsed =classwise_acc* 0.95
    #     print(thrsed)
    #     for i in range(self.num_classes):
    #         pseudo_label[((pseudo_label == i) * (values<thrsed[i]))] = 255
    #
    #     return pseudo_label
    def generate_pseudo_label(self,values, pseudo_label, class_sampling):
        cbst_thresholds = self.generate_class_pseudo_labels_every_iter(values, pseudo_label, 0, class_sampling)
        for j in range(self.num_classes):
            pseudo_label[((pseudo_label == j) * (values < cbst_thresholds[j]))] = 255
        return pseudo_label
    def generate_mask_label(self,label):
        mask_temp = list()
        mask_class = list()

        for i in range(self.num_classes):
            mask_class.append((label == i).to(dtype=torch.float32))

        for i in label:
            if i != 255:

                mask_temp.append(mask_class[i])
            else:
                mask_temp.append(torch.zeros_like(label))
        mask = torch.stack(mask_temp)
        return mask.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #features1 = self.projection_head1(x)
        #features1 = features1.permute(0, 2, 3, 1).reshape(x.shape[0], -1, 256)
        x1 = self.layer5(x,get_feat=True)
        out1 = x1['out']
        # out1 = x1['out']
        features1 = x1['feat']
        # features1 = features1.permute(0, 2, 3, 1).reshape(x.shape[0], -1, 256)
        x2 = self.layer6(x,get_feat=True)
        #features2 = self.projection_head2(x)


        #x2 = self.layer6(x,get_feat=True)
        out2 = x2['out']
        features2 = x2['feat']
        features1 = features1.permute(0, 2, 3, 1).reshape(x.shape[0], -1, 256)
        features2 = features2.permute(0, 2, 3, 1).reshape(x.shape[0], -1, 256)


        return out1,out2,features1, features2


    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []

        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())
        #b.append(self.projection_head1.parameters())
        #b.append(self.projection_head2.parameters())
        # b.append(self.gc1.parameters())
        # b.append(self.gc2.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]


def DeeplabMulti(num_classes=21, use_se = False, train_bn = False, norm_style = 'bn', droprate = 0.1):
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes, use_se = use_se, train_bn = train_bn, norm_style = norm_style, droprate = droprate)
    return model
