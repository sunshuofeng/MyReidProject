
import albumentations as A
from fastai.vision.all import get_image_files
import torch.utils.data as Data
import json
import os.path as osp
from PIL import Image
import numpy as np
import torch
import sys
import copy
import random
from collections import defaultdict





def build_transform(cfg,training=True):
  if training:
    transform=A.Compose([
        A.Resize(cfg['train_size'][0],cfg['train_size'][1]),
        A.RandomBrightnessContrast(),
        A.HorizontalFlip(p=0.5),
        A.Normalize(cfg['mean'],cfg['std'])
    ])
  else:
    transform=A.Compose([
        A.Resize(cfg['test_size'][0],cfg['test_size'][1]),
        A.Normalize(cfg['mean'],cfg['std'])
    ])
  return transform



class PRCC:
    def __init__(self, root='data/prcc_proprecess'):
        self.root = root
        self.train_dir = osp.join(self.root, 'train')
        self.query_dir = osp.join(self.root, 'query')
        self.gallery_dir = osp.join(self.root, 'gallery')
        self.train_sketch_dir = osp.join(self.root, 'train_sketch')
        self.query_sketch_dir = osp.join(self.root, 'query_sketch')
        self.gallery_sketch_dir = osp.join(self.root, 'gallery_sketch')
        with open('train_label.json','r') as f:
            self.train_label=json.load(f)
        train = self._process_dir(self.train_dir)
        query = self._process_dir(self.query_dir)
        gallery = self._process_dir(self.gallery_dir)
        train_sketch = self._process_dir(self.train_sketch_dir)
        query_sketch = self._process_dir(self.query_sketch_dir)
        gallery_sketch = self._process_dir(self.gallery_sketch_dir)

        self.train = train
        self.query = query
        self.gallery = gallery
        self.train_sketch = train_sketch
        self.query_sketch = query_sketch
        self.gallery_sketch = gallery_sketch
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def _process_dir(self, dir_path):
        image_names = get_image_files(dir_path)
        dataset = []
        pid_set = set()
        for name in image_names:
            name = str(name)
            path = name.split('/')[-1]
            pid, cam, _ = path.split('_')
            pid = self.train_label[pid]
            dataset.append((name, pid, cam))
        return dataset


class ImageDataset(Data.Dataset):
    def __init__(self, dataset, data, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = Image.open(img_path)

        path = img_path.split('/')[-1]
        data_type = img_path.split('/')[-2]
        if data_type.startswith('train'):
            sketch_dir = self.data.train_sketch_dir
        elif data_type.startswith('query'):
            sketch_dir = self.data.query_sketch_dir
        else:
            sketch_dir = self.data.gallery_sketch_dir
        sketch_path = osp.join(sketch_dir, path)
        sketch = Image.open(sketch_path)
        image = np.array(img)

        sketch = np.array(sketch)

        transform = self.transform(image=image, mask=sketch)
        image = transform['image']
        sketch = transform['mask']

        image = torch.from_numpy(image.transpose(2, 0, 1))
        sketch = torch.from_numpy(sketch).unsqueeze(0)
        sketch = sketch.repeat(3, 1, 1)

        if int(camid) == 0 or int(camid) == 1:
            date = 0
        else:
            date = 1

        # sketch=sketch.repeat(3,1,1)
        return image, sketch, int(pid), int(camid), date
class DateRandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length

def train_collate_fn1(batch):
    imgs, sketch, pids, camid, date = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), torch.stack(sketch, dim=0), pids


###对于验证集而言，为了提高验证的真实性，我们应该防止同一摄像头的图片进入验证（同一摄像头相当于数据泄露）
def val_collate_fn1(batch):
    imgs, sketch, pids, camids, date = zip(*batch)
    return torch.stack(imgs, dim=0), torch.stack(sketch, dim=0), pids, camids, date


def MAKE_DATALOADER(cfg):
    dataset=PRCC()
    train_transform = build_transform(training=True)
    test_transform = build_transform( training=False)

    data = dataset
    num_classes = data.num_train_pids
    train_set = ImageDataset(data.train, data, train_transform)

    train_loader = Data.DataLoader(train_set, batch_size=cfg['train_bs'],
                                   sampler=DateRandomIdentitySampler(data.train,
                                                                     cfg['train_bs'], cfg['train_K_instances']),
                                   num_workers=cfg['num_workers'],
                                   collate_fn=train_collate_fn1)

    val_set = ImageDataset(data.query + data.gallery, data, test_transform)
    val_loader = Data.DataLoader(
        val_set, batch_size=cfg['test_bs'], shuffle=False, num_workers=cfg['num_workers'],
        collate_fn=val_collate_fn1
    )
    return train_loader, val_loader, len(data.query), num_classes