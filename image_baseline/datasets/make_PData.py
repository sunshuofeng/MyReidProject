import os
import cv2
import numpy as np
import albumentations as A
import shutil
from fastai.vision.all import get_image_files
from collections import defaultdict
import json
from tqdm import tqdm
import random
def make_data(root):
    root,dirs,files=next(os.walk(root))
    print(dirs)
    os.makedirs('data/Pdata_temporary')
    os.makedirs('data/Pdata')
    dst='data/Pdata_temporary'
    pids=set()
    date_id=set()
    date_id_dict={}
    image_p_id=defaultdict(list)
    image_cam_id={}
    image_date_id={}
    with tqdm(total=len(dirs)) as pbar:
        for camid,dir in enumerate(dirs):
            root1,dirs1,files1=next(os.walk(os.path.join(root,dir)))
            for dir1 in dirs1:
                pid=dir1
                pids.add(pid)
                path=os.path.join(root,dir,dir1)
                names=get_image_files(path)
                name_list=[]
                for name in names:
                    name=str(name)
                    name_list.append(name)
                length=len(name_list)
                if length>=5:
                    step=int(length//5)
                    for i in range(0,length-1,step):
                            image_name=name_list[i]
                            name=image_name.split('\\')[-1]
                            name_mem=name.split('_')
                            _,_,date,_=name_mem

                            date_id_dict[date]=len(date_id)
                            date_id.add(date)

                            image_p_id[int(pid)].append(name)
                            image_cam_id[name]=camid
                            image_date_id[name]=date
                            dst_path=os.path.join(dst,name)
                            shutil.copy(image_name,dst_path)
                else:
                    for i in range(0, length - 1, 2):
                        image_name = name_list[i]
                        name = image_name.split('\\')[-1]
                        name_mem = name.split('_')
                        _, _, date, _ = name_mem

                        date_id_dict[date] = len(date_id)
                        date_id.add(date)

                        image_p_id[int(pid)].append(name)
                        image_cam_id[name] = camid
                        image_date_id[name] = date
                        dst_path = os.path.join(dst, name)
                        shutil.copy(image_name, dst_path)
                image_name=name_list[-1]
                name = image_name.split('\\')[-1]
                dst_path = os.path.join(dst, name)
                shutil.copy(image_name, dst_path)
            pbar.update(1)
    print(pids)
    print(date_id_dict)



    file=open('data/Pdata/date_id.json','w')
    json.dump(date_id_dict,file)
    file.close()

    file=open('data/Pdata/image_date.json','w')
    json.dump(image_date_id,file)
    file.close()

    file=open('data/Pdata/image_p.json','w')
    json.dump(image_p_id,file)
    file.close()

    file=open('data/Pdata/image_cam.json','w')
    json.dump(image_cam_id,file)
    file.close()

    pids=list(pids)
    random.shuffle(pids)
    length=len(pids)
    train_len=int(0.5*length)
    train=pids[:train_len]
    test=pids[train_len:]
    with open('data/Pdata/train.txt','w') as f:
        data=','.join(train)
        f.write(data)

    with open('data/Pdata/test.txt','w') as f:
        data=','.join(test)
        f.write(data)

    image_cam_dict={}
    for pid in pids:
        names=image_p_id[int(pid)]
        cam_dict=defaultdict(list)
        for name in  names:
            cam=image_cam_id[name]
            cam_dict[cam].append(name)
        image_cam_dict[int(pid)]=cam_dict

    file=open('data/Pdata/image_cam_dict.json','w')
    json.dump(image_cam_dict,file)
    file.close()

def split_train_test():
    train_ids = []
    test_ids = []
    with open('data/Pdata/train.txt', 'r') as f:
        data = f.read()
        data = data.split(',')
        for id in data:
            train_ids.append(id)
    with open('data/Pdata/test.txt', 'r') as f:
        data = f.read()
        data = data.split(',')
        for id in data:
            test_ids.append(id)

    src_path = 'data/Pdata_temporary'
    dst_path = 'data/Pdata'

    os.makedirs(os.path.join(dst_path, 'train'))
    os.makedirs(os.path.join(dst_path, 'query'))
    os.makedirs(os.path.join(dst_path, 'gallery'))
    file = open('data/Pdata/image_p.json', 'r')
    image_p_dict = json.load(file)
    file.close()

    with tqdm(total=len(train_ids)) as pbar:
        for id in train_ids:
            image_names = image_p_dict[id]
            for name in image_names:
                src = os.path.join(src_path, name)
                dst = os.path.join(dst_path, 'train', name)
                shutil.copy(src, dst)
            pbar.update(1)
    file = open('data/Pdata/image_cam_dict.json', 'r')
    image_cam_dict = json.load(file)
    file.close()
    with tqdm(total=len(test_ids)) as pbar:
        for id in test_ids:
            cams = image_cam_dict[id]
            for key in cams.keys():
                images_names = cams[key]
                random.shuffle(images_names)
                query_name = images_names[0]
                src = os.path.join(src_path, query_name)
                dst = os.path.join(dst_path, 'query', query_name)
                shutil.copy(src, dst)

                gallery_name = images_names[1:]
                for name in gallery_name:
                    src = os.path.join(src_path, name)
                    dst = os.path.join(dst_path, 'gallery', name)
                    shutil.copy(src, dst)
            pbar.update(1)

if __name__ == '__main__':
    import argparse
    parse=argparse.ArgumentParser()
    parse.add_argument('--root',type=str)
    opt=parse.parse_args()
    make_data(opt.root)
    split_train_test()
    shutil.rmtree('data/Pdata_temporary')