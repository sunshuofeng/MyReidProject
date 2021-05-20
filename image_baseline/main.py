import argparse
import torch
import os
import sys
import random
import numpy as np
import logging
from datasets.dataset import *
from models.model import *
from train import *
from utils.scheduler import *
from utils.optimizer import *
from utils.loss import *
import torchvision.transforms as T

def main(cfg):
    ckpt = cfg['ckpt']
    logpt = cfg['logpt']
    checkpoint=Checkpoint(ckpt)
    logger = logging.getLogger()
    fh = logging.FileHandler(logpt)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    train_transform = T.Compose(
        [T.Resize(cfg['train_size']), T.ToTensor(), T.Normalize(mean=cfg['mean'], std=cfg['std'])])
    train_loader, num_classes = Get_Video_train_DataLoader('datasets/mars_database/train_path.txt',
                                                         'datasets/mars_database/train_info.npy', train_transform,num_workers=cfg['num_workers'],
                                                           S=cfg['train_S'],track_per_class=cfg['train_track'],class_per_batch=cfg['train_bs'])
    val_loader = Get_Video_test_DataLoader('datasets/mars_database/test_path.txt', 'datasets/mars_database/test_info.npy',
                                           'datasets/mars_database/query_IDX.npy', train_transform,batch_size=cfg['test_bs'],S=cfg['test_S'])


    model=Baseline(num_classes)
    model=model.cuda()
    criterion=Criterion(cfg,num_classes)
    optimizer=build_optimizer(cfg,model)
    scheduler=build_scheuler(cfg,optimizer,cfg['epochs'],cfg['milestone'])
    train_cfg={}
    train_cfg['train_loader'] = train_loader
    train_cfg['val_loader'] = val_loader
    train_cfg['checkpoint'] = checkpoint
    train_cfg['logger'] = logger
    train_cfg['num_class'] = num_classes
    train_cfg['num_query'] = num_query
    train_cfg['model']=model
    train_cfg['optimizer']=optimizer
    train_cfg['scheduler']=scheduler
    train_cfg['criterion']=criterion
    trainer=Trainer(train_cfg)
    trainer.train(cfg)
    return trainer





if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    parse=argparse.ArgumentParser()


    ##数据定义
    parse.add_argument('--train_bs',type=int,default=4)
    parse.add_argument('--num_workers', type=int, default=0)
    parse.add_argument('--test_bs',type=int,default=4)
    parse.add_argument('--train_S',type=int,default=6)
    parse.add_argument('--test_S',type=int,default=6)
    parse.add_argument('--train_track',type=int,default=8)


    parse.add_argument('--train_size',default=[256,128])
    parse.add_argument('--mean',default=[0.485, 0.456, 0.406])
    parse.add_argument('--std',default=[0.229, 0.224, 0.225])

    ##训练定义
    parse.add_argument('--epochs',type=int,default=50)
    parse.add_argument('--lr',type=int,default=1e-3)
    parse.add_argument('--optimizer',type=str,default='SGD',help='optimizer_name')
    parse.add_argument('--scheduler',type=str,default='Warmup',help='scheduler_name')
    parse.add_argument('--momentum',type=int,default=0.9)
    parse.add_argument('--decay',type=int,default=1e-5)
    parse.add_argument('--metric_learning',default='Triplet')
    parse.add_argument('--margin',default=1.0)
    parse.add_argument('--milestone',default=[20,35])

    parse.add_argument('--ckpt',type=str,default='results/result.pt')
    parse.add_argument('--logpt',type=str,default='results/result.log')
    opt=parse.parse_args()
    cfg=vars(opt)
    main(cfg)