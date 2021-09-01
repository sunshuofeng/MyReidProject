

import argparse
import torch
import os
import sys
import random
import numpy as np
import logging
from datasets.build import make_dataloader
from models.model import *
from train import *
from utils.utils import  *
from utils.scheduler import *
from utils.optimizer import *
from utils.loss import *

def main(cfg):
    ckpt = cfg['ckpt']
    logpt = cfg['logpt']
    checkpoint=Checkpoint(ckpt)
    logger = logging.getLogger()
    fh = logging.FileHandler(logpt)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    train_loaders, val_loader, num_query, num_classes = make_dataloader(cfg)
    model=Model(cfg,num_classes)
    model=model.cuda()
    criterion=Criterion(cfg,num_classes)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['decay'])
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,40)
    train_cfg={}
    train_cfg['train_loaders'] = train_loaders
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
<<<<<<< HEAD
    parse.add_argument('--train_bs',type=int,default=64)
=======
    parse.add_argument('--train_data',type=str,default= 'Market1501')
    parse.add_argument('--val_data',type=str,default= 'Market1501')
    parse.add_argument('--backbone',type=str,default= '50')
    parse.add_argument('--train_bs',type=int,default=4)
>>>>>>> eb770da3269ab82de508d0d37632ad6dbea6ed88
    parse.add_argument('--num_workers', type=int, default=0)
    parse.add_argument('--test_bs',type=int,default=64)
    parse.add_argument('--train_size',default=[256,128])
    parse.add_argument('--mean',default=[0.485, 0.456, 0.406])
    parse.add_argument('--std',default=[0.229, 0.224, 0.225])
    parse.add_argument('--train_K_instances',default=4)
    parse.add_argument('--padding', type=int, default=10)

    ##训练定义
    parse.add_argument('--epochs',type=int,default=120)
    parse.add_argument('--lr',type=int,default=3e-4)
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
