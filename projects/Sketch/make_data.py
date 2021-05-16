from fastai.vision.all import get_image_files
import shutil
import argparse
import os
import json
def make_data(root):
    train_set=set()
    os.makedirs('data/prcc_proprecess')
    os.makedirs('data/prcc_proprecess/train')
    os.makedirs('data/prcc_proprecess/query')
    os.makedirs('data/prcc_proprecess/gallery')
    os.makedirs('data/prcc_proprecess/train_sketch')
    os.makedirs('data/prcc_proprecess/query_sketch')
    os.makedirs('data/prcc_proprecess/gallery_sketch')
    for root, dirs, files in os.walk(os.path.join(root,'rgb','train')):
        for dir in dirs:
            train_set.add(dir)
            path = os.path.join(root, dir)
            image_names = get_image_files(path)
            for i, name in enumerate(image_names):
                name = str(name)
                last_name = name.split('/')[-1]

                cam = last_name[0]
                if cam == 'A':
                    cam = 0
                elif cam == 'B':
                    cam = 1
                elif cam == 'C':
                    cam = 2

                last_name = last_name.split('.')[0][-3:]
                new_name = os.path.join('datasets', 'prcc_proprecess', 'train',
                                        dir + '_' + str(cam) + '_' + last_name + '.jpg')
                shutil.copy(str(name), new_name)
        break

    for root, dirs, files in os.walk(os.path.join(root,'sketch','train')):
        for dir in dirs:
            train_set.add(dir)
            path = os.path.join(root, dir)
            image_names = get_image_files(path)

            for i, name in enumerate(image_names):
                name = str(name)
                last_name = name.split('/')[-1]
                cam = last_name[0]
                if cam == 'A':
                    cam = 0
                elif cam == 'B':
                    cam = 1
                elif cam == 'C':
                    cam = 2
                last_name = last_name.split('.')[0][-3:]
                new_name = os.path.join('datasets', 'prcc_proprecess', 'train_sketch',
                                        dir + '_' + str(cam) + '_' + last_name + '.jpg')
                shutil.copy(str(name), new_name)
        break

    for root, dirs, files in os.walk(os.path.join(root,'rgb','val')):
        for dir in dirs:
            train_set.add(dir)
            path = os.path.join(root, dir)
            image_names = get_image_files(path)
            A_query = True
            B_query = True
            C_query = True
            for i, name in enumerate(image_names):
                name = str(name)
                last_name = name.split('/')[-1]

                if last_name[0] == 'A':
                    if A_query:

                        last_name = last_name.split('.')[0][-3:]
                        new_name = os.path.join('datasets', 'prcc_proprecess', 'query',
                                                dir + '_0' + '_' + last_name + '.jpg')

                        A_query = False
                    else:
                        last_name = last_name.split('.')[0][-3:]
                        new_name = os.path.join('datasets', 'prcc_proprecess', 'gallery',
                                                dir + '_0' + '_' + last_name + '.jpg')

                elif last_name[0] == 'B':
                    if B_query:

                        last_name = last_name.split('.')[0][-3:]
                        new_name = os.path.join('datasets', 'prcc_proprecess', 'query',
                                                dir + '_1' + '_' + last_name + '.jpg')
                        B_query = False
                    else:
                        last_name = last_name.split('.')[0][-3:]
                        new_name = os.path.join('datasets', 'prcc_proprecess', 'gallery',
                                                dir + '_1' + '_' + last_name + '.jpg')

                elif last_name[0] == 'C':
                    if C_query:

                        last_name = last_name.split('.')[0][-3:]
                        new_name = os.path.join('datasets', 'prcc_proprecess', 'query',
                                                dir + '_2' + '_' + last_name + '.jpg')
                        C_query = False
                    else:
                        last_name = last_name.split('.')[0][-3:]
                        new_name = os.path.join('datasets', 'prcc_proprecess', 'gallery',
                                                dir + '_2' + '_' + last_name + '.jpg')
                shutil.copy(name, new_name)

        break

    for root, dirs, files in os.walk(os.path.join(root,'sketch','val')):
        for dir in dirs:
            train_set.add(dir)
            path = os.path.join(root, dir)
            image_names = get_image_files(path)
            A_query = True
            B_query = True
            C_query = True
            for i, name in enumerate(image_names):
                name = str(name)
                last_name = name.split('/')[-1]

                if last_name[0] == 'A':
                    if A_query:

                        last_name = last_name.split('.')[0][-3:]
                        new_name = os.path.join('datasets', 'prcc_proprecess', 'query_sketch',
                                                dir + '_0' + '_' + last_name + '.jpg')
                        A_query = False
                    else:
                        last_name = last_name.split('.')[0][-3:]
                        new_name = os.path.join('datasets', 'prcc_proprecess', 'gallery_sketch',
                                                dir + '_0' + '_' + last_name + '.jpg')

                elif last_name[0] == 'B':
                    if B_query:

                        last_name = last_name.split('.')[0][-3:]
                        new_name = os.path.join('datasets', 'prcc_proprecess', 'query_sketch',
                                                dir + '_1' + '_' + last_name + '.jpg')
                        B_query = False
                    else:
                        last_name = last_name.split('.')[0][-3:]
                        new_name = os.path.join('datasets', 'prcc_proprecess', 'gallery_sketch',
                                                dir + '_1' + '_' + last_name + '.jpg')

                elif last_name[0] == 'C':
                    if C_query:

                        last_name = last_name.split('.')[0][-3:]
                        new_name = os.path.join('datasets', 'prcc_proprecess', 'query_sketch',
                                                dir + '_2' + '_' + last_name + '.jpg')
                        C_query = False
                    else:
                        last_name = last_name.split('.')[0][-3:]
                        new_name = os.path.join('datasets', 'prcc_proprecess', 'gallery_sketch',
                                                dir + '_2' + '_' + last_name + '.jpg')
                shutil.copy(name, new_name)

        break
    train_set = list(train_set)
    train_label = {name: i for i, name in enumerate(train_set)}
    with open('train_label.json', 'w') as f:
        json.dump(train_label, f)



if __name__ == '__main__':

    parse=argparse.ArgumentParser()
    parse.add_argument('--root',type=str)
    opt=parse.parse_args()
    make_data(opt.root)



