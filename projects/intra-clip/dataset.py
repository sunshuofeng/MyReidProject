import os
import sys
import time
import numpy as np
import pandas as pd
import collections
import random
import math
## For torch lib
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
import torch.nn.functional as F
## For Image lib
from PIL import Image
import random
import math
import torchvision.transforms.functional  as TF


class Temporal_flip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        p = random.uniform(0, 1)
        if p > self.p:
            imgs = torch.flip(imgs, dims=[3])
        return imgs


class Temporal_crop:
    def __init__(self, size=(256, 128), p=0.5):
        self.p = p
        self.size = size

    def get_params(self, img, output_size):
        w, h = img.shape[2], img.shape[1]
        th, tw = output_size
        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def __call__(self, imgs):
        p = random.uniform(0, 1)
        if p > self.p:
            img0 = imgs[0]

            i, j, h, w = self.get_params(img0, self.size)
            new_imgs = []
            for img in imgs:
                new_img = TF.crop(img, i, j, h, w)
                new_imgs.append(new_img)
            new_imgs = torch.stack(new_imgs, dim=0)
            imgs = new_imgs
        return imgs


class Temporal_erasing:
    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.p = p
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, imgs):
        p = random.uniform(0, 1)
        if p > self.p:
            img = imgs[0]
            new_imgs = []
            for attempt in range(100):
                area = img.size()[1] * img.size()[2]

                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < img.size()[2] and h < img.size()[1]:
                    x1 = random.randint(0, img.size()[1] - h)
                    y1 = random.randint(0, img.size()[2] - w)
                    for img in imgs:
                        img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                        img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                        img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                        new_imgs.append(img)
                    new_imgs = torch.stack(new_imgs, dim=0)
                    imgs = new_imgs
                    break
        return imgs


def build_temporal_transform():
    transform = []
    transform1 = Temporal_flip()
    transform2 = Temporal_crop()
    transform3 = Temporal_erasing()
    transform.append(transform1)
    transform.append(transform2)
    transform.append(transform3)
    return transform


def process_labels(labels):
    unique_id = np.unique(labels)
    id_count = len(unique_id)
    id_dict = {ID: i for i, ID in enumerate(unique_id.tolist())}
    for i in range(len(labels)):
        labels[i] = id_dict[labels[i]]
    assert len(unique_id) - 1 == np.max(labels)
    return labels, id_count, len(unique_id)





class Video_train_Dataset(Dataset):
    def __init__(self, db_txt, info, transform, S=6, track_per_class=4, flip_p=0.5, delete_one_cam=False,
                 cam_type='normal'):

        with open(db_txt, 'r') as f:
            self.imgs = np.array(f.read().strip().split('\n'))
        self.transform_tempory = build_temporal_transform()
        print(self.imgs)

        if delete_one_cam == True:
            info = np.load(info)

            # 获取行人id以及数量
            info[:, 2], id_count, num_class = process_labels(info[:, 2])

            for i in range(id_count):
                idx = np.where(info[:, 2] == i)[0]

                ##如果这个行人只有一个摄像头拍，那就删除这个行人
                if len(np.unique(info[idx, 3])) == 1:
                    info = np.delete(info, idx, axis=0)
                    id_count -= 1
            info[:, 2], id_count, num_class = process_labels(info[:, 2])

        else:
            info = np.load(info)
            info[:, 2], id_count, num_class = process_labels(info[:, 2])

        self.info = []
        for i in range(len(info)):
            sample_clip = []

            # 获取这段视频有多少帧
            F = info[i][1] - info[i][0] + 1

            # 对视频进行sample,我们希望每个视频抽出6个帧（S参数）作为该视频的序列
            if F < S:

                # 如果该视频帧数小于6帧，则重复最后一帧补全（不能随便抽帧补，不然会破坏时间信息）
                strip = list(range(info[i][0], info[i][1] + 1)) + [info[i][1]] * (S - F)

                # 然后就把帧id放入 后续抽出
                for s in range(S):
                    pool = strip[s * 1:(s + 1) * 1]
                    sample_clip.append(list(pool))
            else:
                # 如果总帧数大于S，通常的做法就是将F分为S组，每组抽一个帧
                interval = math.ceil(F / S)

                ##当然有时候F不是S的整数，就无法每组一样的数量，就需要补全为S的整数
                strip = list(range(info[i][0], info[i][1] + 1)) + [info[i][1]] * (interval * S - F)

                # 但是这里并没有进行抽一个，而是把所有组都放进去，我看了一下，应该是会在后面进行抽取
                for s in range(S):
                    pool = strip[s * interval:(s + 1) * interval]
                    sample_clip.append(list(pool))
            # 添加每个序列的所有帧id，行人id，摄像头id
            self.info.append(np.array([np.array(sample_clip), info[i][2], info[i][3]]))

        self.info = np.array(self.info)
        self.transform = transform
        self.n_id = id_count
        self.n_tracklets = self.info.shape[0]
        self.flip_p = flip_p
        self.track_per_class = track_per_class
        self.cam_type = cam_type
        self.two_cam = False
        self.cross_cam = False
        self.num_class = num_class

    def __getitem__(self, ID):
        # 获取行人的序列集
        sub_info = self.info[self.info[:, 1] == ID]

        if self.cam_type == 'normal':
            # 随机抽取四个序列(self.track_per_class=4)
            tracks_pool = list(np.random.choice(sub_info[:, 0], self.track_per_class))

        elif self.cam_type == 'two_cam':
            ##随机抽取两个摄像头的序列
            unique_cam = np.random.permutation(np.unique(sub_info[:, 2]))[:2]
            tracks_pool = list(np.random.choice(sub_info[sub_info[:, 2] == unique_cam[0], 0], 1)) + \
                          list(np.random.choice(sub_info[sub_info[:, 2] == unique_cam[1], 0], 1))

        elif self.cam_type == 'cross_cam':
            ##想要确保抽取的所有序列之间的摄像头都不一样（当然如果摄像头个数都小于抽取的序列数，那就无法做到了，只能补）
            unique_cam = np.random.permutation(np.unique(sub_info[:, 2]))
            while len(unique_cam) < self.track_per_class:
                unique_cam = np.append(unique_cam, unique_cam)
            unique_cam = unique_cam[:self.track_per_class]
            tracks_pool = []


            for i in range(self.track_per_class):
                tracks_pool += list(np.random.choice(sub_info[sub_info[:, 2] == unique_cam[i], 0], 1))

        one_id_tracks = []

        for track_pool in tracks_pool:
            ##上面说了，如果F>S ，会分为S组，每组抽一个，当然小于S个，补成S个的话，其实就是1个中随机抽1个
            idx = np.random.choice(track_pool.shape[1], track_pool.shape[0])
            number = track_pool[np.arange(len(track_pool)), idx]
            imgs = [self.transform(Image.open(path)) for path in self.imgs[number]]
            imgs = torch.stack(imgs, dim=0)
            for transform in self.transform_tempory:
                imgs = transform(imgs)
            one_id_tracks.append(imgs)

        ##返回的就是序列图片，每个序列的id（与图片重识别不同，每次抽取的都是一个行人的不同序列）
        ##返回(序列数，序列帧数，3，H,W)
        return torch.stack(one_id_tracks, dim=0), ID * torch.ones(self.track_per_class, dtype=torch.int64)

    def __len__(self):
        return self.n_id


def Video_train_collate_fn(data):
    if isinstance(data[0], collections.Mapping):
        t_data = [tuple(d.values()) for d in data]
        values = MARS_collate_fn(t_data)
        return {key: value for key, value in zip(data[0].keys(), values)}
    else:
        imgs, labels = zip(*data)
        imgs = torch.cat(imgs, dim=0)
        labels = torch.cat(labels, dim=0)
        return imgs, labels


def Get_Video_train_DataLoader(db_txt, info, transform, shuffle=True, num_workers=8, S=6, track_per_class=8,
                               class_per_batch=4):
    dataset = Video_train_Dataset(db_txt, info, transform, S, track_per_class)
    dataloader = DataLoader(dataset, batch_size=class_per_batch, collate_fn=Video_train_collate_fn, shuffle=shuffle,
                            worker_init_fn=lambda _: np.random.seed(), drop_last=True, num_workers=num_workers)
    return dataloader, dataset.num_class


class Video_test_Dataset(Dataset):
    def __init__(self, db_txt, info, query, transform, S=6, distractor=True):
        with open(db_txt, 'r') as f:
            self.imgs = np.array(f.read().strip().split('\n'))
        # info
        info = np.load(info)
        self.info = []
        for i in range(len(info)):
            if distractor == False and info[i][2] == 0:
                continue
            sample_clip = []
            F = info[i][1] - info[i][0] + 1
            if F < S:
                strip = list(range(info[i][0], info[i][1] + 1)) + [info[i][1]] * (S - F)
                for s in range(S):
                    pool = strip[s * 1:(s + 1) * 1]
                    sample_clip.append(list(pool))
            else:
                interval = math.ceil(F / S)
                strip = list(range(info[i][0], info[i][1] + 1)) + [info[i][1]] * (interval * S - F)
                for s in range(S):
                    pool = strip[s * interval:(s + 1) * interval]
                    sample_clip.append(list(pool))
            self.info.append(np.array([np.array(sample_clip), info[i][2], info[i][3]]))

        self.info = np.array(self.info)
        self.transform = transform
        self.n_id = len(np.unique(self.info[:, 1]))
        self.n_tracklets = self.info.shape[0]
        self.query_idx = np.load(query).reshape(-1)

        if distractor == False:
            zero = np.where(info[:, 2] == 0)[0]
            self.new_query = []
            for i in self.query_idx:
                if i < zero[0]:
                    self.new_query.append(i)
                elif i <= zero[-1]:
                    continue
                elif i > zero[-1]:
                    self.new_query.append(i - len(zero))
                else:
                    continue
            self.query_idx = np.array(self.new_query)

    def __getitem__(self, idx):
        clips = self.info[idx, 0]
        imgs = [self.transform(Image.open(path)) for path in self.imgs[clips[:, 0]]]
        imgs = torch.stack(imgs, dim=0)
        label = self.info[idx, 1] * torch.ones(1, dtype=torch.int32)
        cam = self.info[idx, 2] * torch.ones(1, dtype=torch.int32)
        return imgs, label, cam

    def __len__(self):
        return len(self.info)


def Video_test_collate_fn(data):
    if isinstance(data[0], collections.Mapping):
        t_data = [tuple(d.values()) for d in data]
        values = MARS_collate_fn(t_data)
        return {key: value for key, value in zip(data[0].keys(), values)}
    else:
        imgs, label, cam = zip(*data)
        imgs = torch.cat(imgs, dim=0)

        labels = torch.cat(label, dim=0)
        cams = torch.cat(cam, dim=0)
        return imgs, labels, cams


def Get_Video_test_DataLoader(db_txt, info, query, transform, batch_size=8, shuffle=False, num_workers=8, S=6,
                              distractor=True):
    dataset = Video_test_Dataset(db_txt, info, query, transform, S, distractor=distractor)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=Video_test_collate_fn, shuffle=shuffle,
                            worker_init_fn=lambda _: np.random.seed(), num_workers=num_workers)
    return dataloader
