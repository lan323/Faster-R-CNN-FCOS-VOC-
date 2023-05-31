# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import beta
import pickle
import torch
import os
#import cv2

def load_labels_name(filename):
    """使用pickle反序列化labels文件，得到存储内容
        cifar10的label文件为“batches.meta”，cifar100则为“meta”
        反序列化之后得到字典对象，可根据key取出相应内容
    """
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def load_data_cifar(filename, mode='cifar10'):
    """ load data and labels information from cifar10 and cifar100
    cifar10 keys(): dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
    cifar100 keys(): dict_keys([b'filenames', b'batch_label', b'fine_labels', b'coarse_labels', b'data'])
    """
    with open(filename, 'rb') as f:
        dataset = pickle.load(f, encoding='bytes')
        if mode == 'cifar10':
            data = dataset[b'data']
            labels = dataset[b'labels']
            img_names = dataset[b'filenames']
        elif mode == 'cifar100':
            data = dataset[b'data']
            labels = dataset[b'fine_labels']
            img_names = dataset[b'filenames']
        else:
            print("mode should be in ['cifar10', 'cifar100']")
            return None, None, None

    return data, labels, img_names


def load_cifar100(cifar100_path, mode='train'):
    if mode == "train":
        filename = os.path.join(cifar100_path, 'train')
        print("Loading {}".format(filename))
        data, labels, img_names = load_data_cifar(filename, mode='cifar100')
    elif mode == "test":
        filename = os.path.join(cifar100_path, 'test')
        print("Loading {}".format(filename))
        data, labels, img_names = load_data_cifar(filename, mode='cifar100')
    else:
        print("mode should be in ['train', 'test']")
        return None, None, None

    return data, labels, img_names


def to_pil(data):
    r = Image.fromarray(data[0])
    g = Image.fromarray(data[1])
    b = Image.fromarray(data[2])
    pil_img = Image.merge('RGB', (r, g, b))
    return pil_img


def random_visualize(imgs, labels, label_names):
    figure = plt.figure(figsize=(len(label_names), 10))
    idxs = list(range(len(imgs)))
    np.random.shuffle(idxs)
    count = [0] * len(label_names)
    for idx in idxs:
        label = labels[idx]
        if count[label] >= 10:
            continue
        if sum(count) > 10 * len(label_names):
            break

        img = to_pil(imgs[idx])
        label_name = label_names[label]

        subplot_idx = count[label] * len(label_names) + label + 1
        plt.subplot(10, len(label_names), subplot_idx)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        if count[label] == 0:
            plt.title(label_name)

        count[label] += 1
    plt.show()

def load_CIFAR_batch(filename):
    with open(filename, 'rb')as f:
        datadict = pickle.load(f,encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'coarse_labels']+datadict[b'fine_labels']
        X = X.reshape(50000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y

def convert_img(imgs):
    img0 = imgs[0]
    img1 = imgs[1]
    img2 = imgs[2]
    i0 = Image.fromarray(img0).convert('L')
    i1 = Image.fromarray(img1).convert('L')
    i2 = Image.fromarray(img2).convert('L')
    img = Image.merge("RGB", (i0, i1, i2))
    return img

def rand_box(img,length): # 随机生成cut区域
    h = img.shape[1]
    w = img.shape[2]
    y = np.random.randint(h)
    x = np.random.randint(w)
    y1 = np.clip(y - length // 2, 0, h)
    y2 = np.clip(y + length // 2, 0, h)
    x1 = np.clip(x - length // 2, 0, w)
    x2 = np.clip(x + length // 2, 0, w)
    return x1, y1, x2, y2

def mixup(img0, img1, lam):
    mixed_img = lam * img0 + (1 - lam) * img1
    return mixed_img

def cutout(img,randbox):
    mask = np.ones((img.shape[1], img.shape[2]), np.float32)
    x1, y1, x2, y2 = randbox[0],randbox[1],randbox[2],randbox[3]
    mask[x1: x2, y1: y2] = 0.
    img = img * mask
    return img

def cutmix(img0, img1, randbox, lam):
    img = img0.copy()
    bbx1, bby1, bbx2, bby2 = randbox[0],randbox[1],randbox[2],randbox[3]
    img[:,bbx1:bbx2, bby1:bby2] = img1[:, bbx1:bbx2, bby1:bby2]
    return img

if __name__ == '__main__':
    # python imc/train.py --method baseline --dataset cifar100 --model resnet18 --epochs 200
    # python imc/train.py --method cutout --dataset cifar100 --model resnet18 --data_augmentation --epochs 200
    # python imc/train.py --method mixup --dataset cifar100 --model resnet18 --data_augmentation --epochs 200
    # python imc/train.py --method cutmix --dataset cifar100 --model resnet18 --data_augmentation --epochs 200  --cutmix_prob 0.1

    cifar100_path = r"C:\Users\admin\PycharmProjects\pythonProject3\data\cifar-100-python"
    obj_cifar100 = load_labels_name(os.path.join(cifar100_path, 'meta'))

    data_cifar100_train, labels_cifar100_train, img_names_cifar100_train = load_cifar100(cifar100_path, mode='train')
    data_cifar100_test, labels_cifar100_test, img_names_cifar100_test = load_cifar100(cifar100_path, mode='test')
    imgs_cifar100_train = data_cifar100_train.reshape(data_cifar100_train.shape[0], 3, 32, 32)
    imgs_cifar100_test = data_cifar100_test.reshape(data_cifar100_test.shape[0], 3, 32, 32)

    label_names_cifar100 = obj_cifar100['fine_label_names']
    #random_visualize(imgs=imgs_cifar100_train, labels=labels_cifar100_train, label_names=label_names_cifar100)

    imgX, imgY = load_CIFAR_batch(cifar100_path + "\\train")
    print(imgX.shape)

    N = 3
    length = 16
    lam = 0.7

    for idx in range(N):
        i, j = np.random.randint(0, len(imgX), 2)
        train_img = imgX[i]
        rand_img = imgX[j]
        randbox = rand_box(train_img, length)

        img_cutout = cutout(train_img, randbox)
        img_mixup = mixup(train_img, rand_img, lam)
        img_cutmix = cutmix(train_img, rand_img, randbox, lam)

        plt.subplot(N, 5, idx * 5 + 1)
        img2 = convert_img(train_img)
        plt.imshow(img2)
        plt.title('img')

        plt.subplot(N, 5, idx * 5 + 2)
        img_cutout2 = convert_img(img_cutout)
        plt.imshow(img_cutout2)
        plt.title('cutout')

        plt.subplot(N, 5, idx * 5 + 3)
        img_mixup2 = convert_img(img_mixup)
        plt.imshow(img_mixup2)
        plt.title('mixup')

        plt.subplot(N, 5, idx * 5 + 4)
        img_cutmix2 = convert_img(img_cutmix)
        plt.imshow(img_cutmix2)
        plt.title('cutmix')

        plt.subplot(N, 5, idx * 5 + 5)
        rand_img2 = convert_img(rand_img)
        plt.imshow(rand_img2)
        plt.title('img_mix')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    plt.savefig(cifar100_path + "_aug_vis.png")
    plt.show()