import glob
import re
import os.path as osp
import cv2
import torch

class Market1501():
    def __init__(self, root='/content/datasets', verbose=True, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self._check_before_run()
        self.date_lentg=1
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)
        self.train = train
        self.query = query
        self.gallery = gallery
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def print_dataset_statistics(self, train, query, gallery):
            num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
            num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
            num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)
            print("Dataset statistics:")
            print("  ----------------------------------------")
            print("  subset   | # ids | # images | # cameras")
            print("  ----------------------------------------")
            print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
            print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
            print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
            print("  ----------------------------------------")


    def get_imagedata_info(self, data):
            pids, cams = [], []
            for _, pid, camid, _ in data:
                pids += [pid]
                cams += [camid]
            pids = set(pids)
            cams = set(cams)
            num_pids = len(pids)
            num_cams = len(cams)
            num_imgs = len(data)
            return num_pids, num_imgs, num_cams

    def _check_before_run(self):
            """Check if all files are available before going deeper"""
            if not osp.exists(self.dataset_dir):
                raise RuntimeError("'{}' is not available".format(self.dataset_dir))
            if not osp.exists(self.train_dir):
                raise RuntimeError("'{}' is not available".format(self.train_dir))
            if not osp.exists(self.query_dir):
                raise RuntimeError("'{}' is not available".format(self.query_dir))
            if not osp.exists(self.gallery_dir):
                raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
            pattern = re.compile(r'([-\d]+)_c(\d)')
            pid_container = set()
            for img_path in img_paths:
                pid, _ = map(int, pattern.search(img_path).groups())
                if pid == -1: continue  # junk images are just ignored
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
            dataset = []
            for img_path in img_paths:
                pid, camid = map(int, pattern.search(img_path).groups())
                if pid == -1: continue  # junk images are just ignored
                assert 0 <= pid <= 1501  # pid == 0 means background
                assert 1 <= camid <= 6
                camid -= 1  # index starts from 0
                if relabel: pid = pid2label[pid]
                dataset.append((image_path,pid, camid,1))
            return dataset


from fastai.vision.all import get_image_files
import torch.utils.data as Data
import json
import os

class PDataset:
    def __init__(self, root):
        self.root = root
        self.train_dir = osp.join(self.root, 'train')
        self.query_dir = osp.join(self.root, 'query')
        self.gallery_dir = osp.join(self.root, 'gallery')
        self.date_length=11
        train = self._process_dir(self.train_dir)
        query = self._process_dir(self.query_dir)
        gallery = self._process_dir(self.gallery_dir)
        self.train = train
        self.query = query
        self.gallery = gallery
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
        file1 = open(os.path.join(root,'image_cam.json'), 'r')
        self.image_cams = json.load(file1)
        file1.close()

        file1 = open(os.path.join(root,'image_cam_dict.json'), 'r')
        self.image_cam_dict = json.load(file1)
        file1.close()

        file1 = open(os.path.join(root,'image_p.json'), 'r')
        self.image_p = json.load(file1)
        file1.close()

        file1 = open(os.path.join(root,'date_id.json'), 'r')
        self.date_ids = json.load(file1)
        file1.close()

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid, _ in data:
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
            pid, _, date, _ = path.split('_')
            pid_set.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_set)}
        for name in image_names:
            name = str(name)
            path = name.split('/')[-1]
            pid, _, date, _ = path.split('_')
            pid = pid2label[pid]
            cam = self.image_cams[path]
            date = self.date_ids[date]
            dataset.append((name, int(pid), int(cam), int(date)))
        return dataset


import glob
import re
import os.path as osp


class VC_Clothes():
    def __init__(self, root='/content/datasets', verbose=True, **kwargs):
        super(VC_Clothes, self).__init__()
        self.dataset_dir = root

        train = self._process_dir(self.dataset_dir, relabel=True)
        query = self._process_dir(self.dataset_dir, relabel=False)
        gallery = self._process_dir(self.dataset_dir, relabel=False)

        if verbose:
            print("=> VC loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid, _ in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    ##最终返回的是每张图片的路径，pid，camid
    def _process_dir(self, dir_path, relabel=False):
        image_names = get_image_files(dir_path)

        pid_container = set()
        for name in image_names:
            name = str(name)
            path = name.split('/')[-1]
            pid, date, _, _ = path.split('-')
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for name in image_names:
            name = str(name)
            path = name.split('/')[-1]
            pid, date, _, _ = path.split('-')
            if relabel: pid = pid2label[pid]
            dataset.append((name, pid, 1, 1))
        return dataset




class Real28():
    def __init__(self, root='/content/datasets', verbose=True, **kwargs):
        super(Real28, self).__init__()
        self.dataset_dir = root
        train = self._process_dir(self.dataset_dir, relabel=True)
        query = self._process_dir(self.dataset_dir, relabel=False)
        gallery = self._process_dir(self.dataset_dir, relabel=False)

        if verbose:
            print("=> real loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid, _ in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    ##最终返回的是每张图片的路径，pid，camid
    def _process_dir(self, dir_path, relabel=False):
        image_names = get_image_files(dir_path)

        pid_container = set()
        for name in image_names:
            name = str(name)
            path = name.split('/')[-1]
            pid, date, _, _ = path.split('_')
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for name in image_names:
            name = str(name)
            path = name.split('/')[-1]
            pid, date, _, _ = path.split('_')
            if relabel: pid = pid2label[pid]
            dataset.append((name, pid, 1, 1))
        return dataset




class Duke():
    def __init__(self, root='/kaggle/input/dukemtmcreid', verbose=True, **kwargs):
        super(Duke, self).__init__()
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Duke loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid,_ in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))


        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid,1))

        return dataset




import torch.utils.data as Data
from PIL import Image




class ImageDataset(Data.Dataset):
    def __init__(self, dataset, transform=None, pid_add=0):
        self.dataset = dataset
        self.transform = transform
        self.pid_add = pid_add

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid,date = self.dataset[index]
        img = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(img)

        return image, pid + self.pid_add, camid, img_path,date
