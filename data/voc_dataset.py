import os
import os.path as osp
import numpy as np
import random
#import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
from torch.utils import data
from PIL import Image,ImageFilter
#import matplotlib.pyplot as plt
from torchvision import transforms
import PIL.Image
class VOCDataSet(data.Dataset):
    def __init__(self, root, list_path, crop_size=(321, 321), mean=(0.485, 0.456, 0.406), scale=False, mirror=False,augment=False,flip=False, ignore_label=255,split='train'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        if self.crop_size!=None:
            self.crop_size = crop_size
            self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.image_padding = (np.array(mean) * 255.).tolist()
        self.mean = mean
        self.is_mirror = mirror
        self.augment = augment
        self.split = split
        if self.augment == augment:
            self.flip = flip

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        # if not max_iters==None:
	    #     self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
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
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)

        image -= self.mean
        if 'val' not in self.split:
            img_h, img_w = label.shape
            pad_h = max(self.crop_h - img_h, 0)
            pad_w = max(self.crop_w - img_w, 0)
            if pad_h > 0 or pad_w > 0:
                img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                             pad_w, cv2.BORDER_CONSTANT,
                                             value=(0.0, 0.0, 0.0))
                label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                               pad_w, cv2.BORDER_CONSTANT,
                                               value=(self.ignore_label,))
            else:
                img_pad, label_pad = image, label

            img_h, img_w = label_pad.shape
            h_off = random.randint(0, img_h - self.crop_h)
            w_off = random.randint(0, img_w - self.crop_w)
            image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
            label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        # if self.augment:
        #     image_wk, image_str, label = self._augmentation(image, label)
        #     image_wk = np.asarray(image_wk)
        #     image_str = np.asarray(image_str)
        #     label = np.asarray(label)
        #     return image_wk.copy(), image_str.copy(), label.copy(), np.array(size), name
        # else:
        #     if self.scale:
        #         image, label = self.generate_scale_label(image, label)
        #     if self.crop_size is not None:
        #         image, label = self._crop(image, label)
        #     if self.flip:
        #         image, label = self._flip(image, label)
        #     mean = [0.485, 0.456, 0.406]
        #     std = [0.229, 0.224, 0.225]
        #     normalize = transforms.Normalize(mean, std)
        #     to_tensor = transforms.ToTensor()
        #     image = normalize(to_tensor(image))
        #     image = np.asarray(image, np.float32)
        #image = image.transpose((2, 0, 1))
        return image.copy(), label.copy(), np.array(size), name

class VOC12ImageDataset(data.Dataset):

    def __init__(self, root, list_path,  transform=None):
        self.list_path = list_path
        # self.img_name_list = load_img_name_list(img_name_list_path)
        self.root = root
        self.transform = transform
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        # if not max_iters==None:
        #     self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]


        # image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        image = PIL.Image.open(datafiles["img"]).convert("RGB")
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        name = datafiles["name"]
        if self.transform:
            image = self.transform(image)

        return name, image
class VOCPDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(0.485, 0.456, 0.406), scale=True, mirror=False,augment=False,flip=False, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        if self.crop_size!=None:
            self.crop_size = crop_size
            self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.image_padding = (np.array(mean) * 255.).tolist()
        self.mean = mean
        self.is_mirror = mirror
        self.augment = augment
        if self.augment == augment:
            self.flip = flip
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        # if not max_iters==None:
	    #     self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "8_pseudo/%s.npy" % name)
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
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = np.load(datafiles['label'])
        #print(label)
        #label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        # label = Image.open(datafiles["label"])
        # label = np.asarray(label, np.float32)
        # label = label.transpose((2, 0, 1))
        # label = label[0]
        #label = np.asarray(label, np.uint8)
        # image, label = Image.open(datafiles["img"]).convert('RGB'), Image.open(datafiles["label"])
        # image, label = np.asarray(image, np.float32), np.asarray(label, np.uint8)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)



        #image = np.asarray(image, np.float32)

        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        # if self.augment:
        #     image_wk, image_str, label = self._augmentation(image, label)
        #     image_wk = np.asarray(image_wk)
        #     image_str = np.asarray(image_str)
        #     label = np.asarray(label)
        #     return image_wk.copy(), image_str.copy(), label.copy(), np.array(size), name
        # else:
        #     if self.scale:
        #         image, label = self.generate_scale_label(image, label)
        #     if self.crop_size is not None:
        #         image, label = self._crop(image, label)
        #     if self.flip:
        #         image, label = self._flip(image, label)
        #     mean = [0.485, 0.456, 0.406]
        #     std = [0.229, 0.224, 0.225]
        #     normalize = transforms.Normalize(mean, std)
        #     to_tensor = transforms.ToTensor()
        #     image = normalize(to_tensor(image))
        #     image = np.asarray(image, np.float32)
        #     #image = image.transpose((2, 0, 1))
        return image.copy(), label.copy(), np.array(size), name



class VOCGTDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
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

    def __getitem__(self, index):

        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        # label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        label = Image.open(datafiles["label"])
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.uint8)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
        return image.copy(), label.copy(), np.array(size), name

class VOCDataTestSet(data.Dataset):
    def __init__(self, root, list_path, crop_size=(505, 505), mean=(128, 128, 128)):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            self.files.append({
                "img": img_file
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        size = image.shape
        name = osp.splitext(osp.basename(datafiles["img"]))[0]
        image = np.asarray(image, np.float32)
        image -= self.mean

        img_h, img_w, _ = image.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0))
        image = image.transpose((2, 0, 1))

        return image, name, size


if __name__ == '__main__':
    DATA_DIRECTORY = 'D:\HHZ\Seg\data\\voc_dataset\\'
    DATA_LIST_PATH = 'D:\HHZ\Seg\data\\voc_list\\train_aug.txt'
    dst = VOCDataSet(DATA_DIRECTORY,DATA_LIST_PATH,crop_size=(321,321),scale=True, mirror=False,mean=(0.485, 0.456, 0.406),augment=True, flip=True)
    trainloader = data.DataLoader(dst, batch_size=4, shuffle=False)
    for i, data in enumerate(trainloader):
        imgs_w,imgs_s, labels,_,_ = data

        if i == 0:
            #imgs = torchvision.utils.make_grid(imgs_w).numpy()
            print(imgs_w.shape)
            print(labels.shape)
            #imgs = np.transpose(imgs_w, (0, 2, 0))
            #imgs = imgs[:, :, ::-1]
            imgs = Image.fromarray(np.uint8(imgs_s[1]))
            imgs.save('voc_Demo.png')
            labels = torchvision.utils.make_grid(labels).numpy()
            labels = np.transpose(labels, (1, 2, 0))
            labels = labels[:, :, ::-1]
            labels = Image.fromarray(np.uint8(labels))
            labels.save('voc_lab.png')
            #plt.imshow(img)
            #plt.show()
        break