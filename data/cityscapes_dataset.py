import os
import os.path as osp
import numpy as np
import random
#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image, ImageFile
#from dataset.autoaugment import ImageNetPolicy
from torchvision import transforms
import time
import cv2
from PIL import Image,ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True

class cityscapesDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(256, 512), mean=(128, 128, 128), scale=False, mirror=True, set='val', ignore_label=255, augment= False, flip=False):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.image_padding = (np.array(mean) * 255.).tolist()
        self.is_mirror = mirror
        self.augment = augment
        if self.augment == augment:
            self.flip = flip
        if self.crop_size!=None:
            self.h = self.crop_size[0]
            self.w = self.crop_size[1]
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        self.set = set
        # for split in ["train", "trainval", "val"]:
         
        #https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            #img_file = osp.join(self.root)
            label_file = osp.join(self.root, "gtFine/%s/%s" % (self.set, name.replace('leftImg8bit', 'gtFine_labelIds') ))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)
    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label
    def _crop(self, image, label):
        # Padding to return the correct crop size
        if (isinstance(self.crop_size, list) or isinstance(self.crop_size, tuple)) and len(self.crop_size) == 2:
            crop_h, crop_w = self.crop_size
        elif isinstance(self.crop_size, int):
            crop_h, crop_w = self.crop_size, self.crop_size
        else:
            raise ValueError

        h, w, _ = image.shape
        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT, }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=self.image_padding, **pad_kwargs)
            label = cv2.copyMakeBorder(label, value=self.ignore_label, **pad_kwargs)

        # Cropping
        h, w, _ = image.shape
        start_h = random.randint(0, h - crop_h)
        start_w = random.randint(0, w - crop_w)
        end_h = start_h + crop_h
        end_w = start_w + crop_w
        image = image[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]
        return image, label

    def data_aug(self,images):
        kernel_size = int(random.random() * 4.95)
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        blurring_image = transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
        color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean, std)
        to_tensor = transforms.ToTensor()
        weak_aug = normalize(to_tensor(images.copy()))
        strong_aug = images

        if random.random() < 0.8:
            strong_aug = color_jitter(strong_aug)
        strong_aug = transforms.RandomGrayscale(p=0.2)(strong_aug)

        if random.random() < 0.5:
            sigma = np.random.uniform(0.1, 2.0)
            strong_aug = strong_aug.filter(ImageFilter.GaussianBlur(radius=sigma))
            #strong_aug = blurring_image(strong_aug)
        strong_aug = normalize(to_tensor(strong_aug))


        return weak_aug, strong_aug


    def _flip(self, image, label):
        # Random H flip
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()
        return image, label

    def _augmentation(self, image, label):
        h, w, _ = image.shape
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        if self.crop_size is not None:
            image, label = self._crop(image, label)
        if self.flip:
            image, label = self._flip(image, label)

        image = Image.fromarray(np.uint8(image))

        image_wk, image_str = self.data_aug(image)
        return image_wk, image_str, label

    def __getitem__(self, index):
        #tt = time.time()
        datafiles = self.files[index]
        name = datafiles["name"]

        image, label = Image.open(datafiles["img"]).convert('RGB'), Image.open(datafiles["label"])
        image, label = np.asarray(image, np.float32), np.asarray(label, np.uint8)
        size = image.shape
        # label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
        # for k, v in list(self.id_to_trainid.items()):
        #     label_copy[label == k] = v
        #
        # resize
        #image, label = image.resize(self.resize_size, Image.BICUBIC), label.resize(self.resize_size, Image.NEAREST)


        # if self.autoaug:
        #     policy = ImageNetPolicy()
        #     image = policy(image)

        #image, label = np.asarray(image, np.float32), np.asarray(label, np.uint8)
        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
        for k, v in list(self.id_to_trainid.items()):
            label_copy[label == k] = v

        size = image.shape
        #print(size)
        # image = image[:, :, ::-1]  # change to BGR
        # image -= self.mean
        # image = image.transpose((2, 0, 1))
        # x1 = random.randint(0, image.shape[1] - self.h)
        #
        #
        # y1 = random.randint(0, image.shape[2] - self.w)
        #
        # image = image[:, x1:x1+self.h, y1:y1+self.w]
        # #print(image.shape)
        # label_copy = label_copy[x1:x1+self.h, y1:y1+self.w]
        #
        # if self.is_mirror and random.random() < 0.5:
        #     image = np.flip(image, axis = 2)
        #     label_copy = np.flip(label_copy, axis = 1)
        # #print('Time used: {} sec'.format(time.time()-tt))
        # return image.copy(), label_copy.copy(), np.array(size), name

        if self.augment:
            image_wk, image_str, label = self._augmentation(image, label_copy)
            image_wk = np.asarray(image_wk)
            image_str = np.asarray(image_str)
            label = np.asarray(label)
            return image_wk.copy(), image_str.copy(), label.copy(), np.array(size), name,index
        else:
            if self.scale:
                image, label_copy = self.generate_scale_label(image, label_copy)
            if self.crop_size is not None:
                image, label_copy = self._crop(image, label_copy)
            if self.flip:
                image, label_copy = self._flip(image, label_copy)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            normalize = transforms.Normalize(mean, std)
            to_tensor = transforms.ToTensor()
            image = normalize(to_tensor(image))
            image = np.asarray(image, np.float32)
            #image = image.transpose((2, 0, 1))
            return image.copy(), label_copy.copy(), np.array(size), name


class cityscapesPDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(256, 512), mean=(128, 128, 128), scale=False,
                 mirror=True, set='val', ignore_label=255, augment=False, flip=False):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.image_padding = (np.array(mean) * 255.).tolist()
        self.is_mirror = mirror
        self.augment = augment
        if self.augment == augment:
            self.flip = flip
        self.h = self.crop_size[0]
        self.w = self.crop_size[1]
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        self.set = set
        # for split in ["train", "trainval", "val"]:

        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        self.trainid2name = {
            0: "road",
            1: "sidewalk",
            2: "building",
            3: "wall",
            4: "fence",
            5: "pole",
            6: "light",
            7: "sign",
            8: "vegetation",
            9: "terrain",
            10: "sky",
            11: "person",
            12: "rider",
            13: "car",
            14: "truck",
            15: "bus",
            16: "train",
            17: "motocycle",
            18: "bicycle",
        }
        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            # img_file = osp.join(self.root)
            label_file = osp.join(self.root,
                                  "30_0.4_pseudo/%s/%s" % (self.set, name))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label

    def _crop(self, image, label):
        # Padding to return the correct crop size
        if (isinstance(self.crop_size, list) or isinstance(self.crop_size, tuple)) and len(self.crop_size) == 2:
            crop_h, crop_w = self.crop_size
        elif isinstance(self.crop_size, int):
            crop_h, crop_w = self.crop_size, self.crop_size
        else:
            raise ValueError

        h, w, _ = image.shape
        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT, }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=self.image_padding, **pad_kwargs)
            label = cv2.copyMakeBorder(label, value=self.ignore_label, **pad_kwargs)

        # Cropping
        h, w, _ = image.shape
        start_h = random.randint(0, h - crop_h)
        start_w = random.randint(0, w - crop_w)
        end_h = start_h + crop_h
        end_w = start_w + crop_w
        image = image[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]
        return image, label

    def data_aug(self, images):
        kernel_size = int(random.random() * 4.95)
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        blurring_image = transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
        color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean, std)
        to_tensor = transforms.ToTensor()
        weak_aug = normalize(to_tensor(images.copy()))
        strong_aug = images

        if random.random() < 0.8:
            strong_aug = color_jitter(strong_aug)
        strong_aug = transforms.RandomGrayscale(p=0.2)(strong_aug)

        if random.random() < 0.5:
            sigma = np.random.uniform(0.1, 2.0)
            strong_aug = strong_aug.filter(ImageFilter.GaussianBlur(radius=sigma))
            # strong_aug = blurring_image(strong_aug)
        strong_aug = normalize(to_tensor(strong_aug))

        return weak_aug, strong_aug

    def _flip(self, image, label):
        # Random H flip
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()
        return image, label

    def _augmentation(self, image, label):
        h, w, _ = image.shape
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        if self.crop_size is not None:
            image, label = self._crop(image, label)
        if self.flip:
            image, label = self._flip(image, label)

        image = Image.fromarray(np.uint8(image))

        image_wk, image_str = self.data_aug(image)
        return image_wk, image_str, label

    def __getitem__(self, index):
        # tt = time.time()
        datafiles = self.files[index]
        name = datafiles["name"]

        image, label = Image.open(datafiles["img"]).convert('RGB'), Image.open(datafiles["label"])
        image, label = np.asarray(image, np.float32), np.asarray(label, np.float32)

        size = image.shape
        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
        # for k, v in list(self.id_to_trainid.items()):
        #     label_copy[label == k] = v
        for k in self.trainid2name.keys():
            label_copy[label == k] = k
        #label_copy = Image.fromarray(label_copy)

        # resize
        #image, label = image.resize(self.resize_size, Image.BICUBIC), label.resize(self.resize_size, Image.NEAREST)


        # if self.autoaug:
        #     policy = ImageNetPolicy()
        #     image = policy(image)


        #print(size)
        # image = image[:, :, ::-1]  # change to BGR
        # image -= self.mean
        # image = image.transpose((2, 0, 1))
        # x1 = random.randint(0, image.shape[1] - self.h)
        #
        #
        # y1 = random.randint(0, image.shape[2] - self.w)
        #
        # image = image[:, x1:x1+self.h, y1:y1+self.w]
        # #print(image.shape)
        # label_copy = label_copy[x1:x1+self.h, y1:y1+self.w]
        #
        # if self.is_mirror and random.random() < 0.5:
        #     image = np.flip(image, axis = 2)
        #     label_copy = np.flip(label_copy, axis = 1)
        # #print('Time used: {} sec'.format(time.time()-tt))
        # return image.copy(), label_copy.copy(), np.array(size), name

        if self.augment:
            image_wk, image_str, label = self._augmentation(image, label_copy)
            image_wk = np.asarray(image_wk)
            image_str = np.asarray(image_str)
            label = np.asarray(label)
            return image_wk.copy(), image_str.copy(), label.copy(), np.array(size), name, index
        else:
            if self.scale:
                image, label_copy = self.generate_scale_label(image, label_copy)
            if self.crop_size is not None:
                image, label_copy = self._crop(image, label_copy)
            if self.flip:
                image, label_copy = self._flip(image, label_copy)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            normalize = transforms.Normalize(mean, std)
            to_tensor = transforms.ToTensor()
            image = normalize(to_tensor(image))
            image = np.asarray(image, np.float32)
            # image = image.transpose((2, 0, 1))
            return image.copy(), label_copy.copy(), np.array(size), name

if __name__ == '__main__':
    dst = cityscapesDataSet('./data/Cityscapes/data', './cityscapes_list/train.txt',

                                                     crop_size=(512,512),
                                                     scale=False, mirror=False, mean=np.array((0.485, 0.456, 0.406), dtype=np.float32),
                                                     augment=True, set = 'train',flip=True)
    trainloader = data.DataLoader(dst, batch_size=2,shuffle=True,pin_memory=True, drop_last=True)
    for i, data in enumerate(trainloader):
        imgs, label, _, _,_= data
        if i == 0:
            imgs = torchvision.utils.make_grid(imgs).numpy()
            imgs = np.transpose(imgs, (1, 2, 0))
            imgs = imgs[:, :, ::-1]
            imgs = Image.fromarray(np.uint8(imgs) )
            imgs.save('Cityscape_Demo1.png')
            label = torchvision.utils.make_grid(label).numpy()
            label = np.transpose(label, (1, 2, 0))
            label = label[:, :, ::-1]
            label = Image.fromarray(np.uint8(label))
            label.save('Cityscape_label1.png')
        break
