import argparse
import os
import numpy as np
import scipy.io as sio

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    path=os.path.abspath('..')
    parser.add_argument('--data_dir',help='path/to/MARS/',default=os.path.join(path,'MARS-v160809'))
    parser.add_argument('--info_dir',help='path/to/MARS-evaluation/info/',default=os.path.join(path,'MARS-v160809','MARS-evaluation-master','info'))
    parser.add_argument('--output_dir',help='path/to/save/database',default='mars_database')
    args = parser.parse_args()
    os.makedirs('mars_database')

    # os.system('mkdir -p %s'%(args.output_dir))
    # Train
    train_imgs = []
    data_dir = os.path.join(args.data_dir,'bbox_train')
    #获取所有的id
    ids = sorted(os.listdir(data_dir))
    for id in ids:
        #获取id下的所有图片
        images = sorted(os.listdir(os.path.join(data_dir,id)))
        for image in images:
            if is_image_file(image):
                #添加图片的路径
                train_imgs.append(os.path.abspath(os.path.join(data_dir,id,image)))
    train_imgs = np.array(train_imgs)
    #存到txt中
    np.savetxt(os.path.join(args.output_dir,'train_path.txt'),train_imgs,fmt='%s',delimiter='\n')

    # Test与上面同理
    test_imgs = []
    data_dir = os.path.join(args.data_dir,'bbox_test')
    ids = sorted(os.listdir(data_dir))
    for id in ids:
        images = sorted(os.listdir(os.path.join(data_dir,id)))
        for image in images:
            if is_image_file(image):
                test_imgs.append(os.path.abspath(os.path.join(data_dir,id,image)))
    test_imgs = np.array(test_imgs)
    np.savetxt(os.path.join(args.output_dir,'test_path.txt'),test_imgs,fmt='%s',delimiter='\n')

    ## process matfile
    train_info = sio.loadmat(os.path.join(args.info_dir,'tracks_train_info.mat'))['track_train_info']
    test_info = sio.loadmat(os.path.join(args.info_dir,'tracks_test_info.mat'))['track_test_info']
    query_IDX = sio.loadmat(os.path.join(args.info_dir,'query_IDX.mat'))['query_IDX']

    # start from 0 (matlab starts from 1)
    train_info[:,0:2] = train_info[:,0:2]-1
    test_info[:,0:2] = test_info[:,0:2]-1
    query_IDX = query_IDX -1
    np.save(os.path.join(args.output_dir,'train_info.npy'),train_info)
    np.save(os.path.join(args.output_dir,'test_info.npy'),test_info)
    np.save(os.path.join(args.output_dir,'query_IDX.npy'),query_IDX)
    
