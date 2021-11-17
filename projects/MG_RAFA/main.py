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
import torch.nn as nn
from torch.nn import init


from torch.nn import functional as F
from torchvision.models import resnet50
import torch
import torch.nn as nn
from data import *
from utils import *
from dataclasses import dataclass

from backbone import *

class Model(nn.Module):
    def __init__(self, cfg, num_feature, D):
        super(Model, self).__init__()
        self.relu = nn.ReLU()

        ##论文使用的是1*1卷积，实质上和全连接层一模一样，因为本质上输入是二维的,K个特征节点，每个节点是C维特征向量
        self.W_u = nn.Conv2d(num_feature, int(num_feature / cfg.s),kernel_size=1)
        self.bn_u = nn.BatchNorm2d(int(num_feature /  cfg.s))

        self.W_v = nn.Conv2d(num_feature, int(num_feature /  cfg.s),kernel_size=1)
        self.bn_v = nn.BatchNorm2d(int(num_feature /  cfg.s))

        self.W_phi = nn.Conv2d(num_feature, int(num_feature /  cfg.s),kernel_size=1)
        self.bn_phi = nn.BatchNorm2d(int(num_feature /  cfg.s))

        self.W_psi = nn.Conv1d(D, int(D /  cfg.s),kernel_size=1)
        self.bn_psi = nn.BatchNorm1d(int(D /  cfg.s))

        self.W_theta = nn.Conv1d(int(D / cfg.s) + int(num_feature /  cfg.s), num_feature,kernel_size=1)
        self.bn_theta = nn.BatchNorm1d(num_feature)


    ##这里我假设T=8
    def forward(self, x):
        B, C, H, W = x.shape
        
        b = int(B / 8)
        T = 8

        ##普通特征节点
        F_all=x

        ##平均特征节点
        F_R=x.view(b,T,C,H,W)
        F_R=torch.mean(F_R,dim=1,keepdim=True).view(b,C,H,W)
        F_all_u = self.relu(self.bn_u(self.W_u(F_all)))
        F_R_v = self.relu(self.bn_v(self.W_v(F_R)))
        c=F_all_u.shape[1]

        
        F_all_u = F_all_u.view(b, T, C,-1).permute(0,1,3,2).contiguous().view(b,-1,c)
        F_R_v = F_R_v.view(b,c,-1).permute(0,2,1).contiguous()
        
        # print("all:{}".format(F_all_u.shape))
        # print("R:{}".format(F_R_v.shape))
        # print("x:{}".format(x.shape))
         ##点乘操作
        ##b,T*H*W,H*W
        Relation = torch.matmul(F_all_u, F_R_v.permute(0,2,1))
        # print("Rela:{}".format(Relation.shape))
        Relation = Relation.view(b,-1,H*W).permute(0,2,1).contiguous()
        

        # print(Relation.shape)
        x1=self.relu(self.bn_phi(self.W_phi(F_all)))
        c=x1.shape[1]
        x1=x1.view(b,T,c,-1).permute(0,2,1,3).contiguous().view(b,c,-1)



        Relation = self.relu(self.bn_psi(self.W_psi(Relation)))
        

        
        A = self.relu(self.bn_theta(self.W_theta(torch.cat([x1, Relation], dim=1))))
        A = F.softmax(A,dim=-1)
        F_all=F_all.view(b,T,C,H*W).permute(0,2,1,3).contiguous().view(b,C,-1)
        V = torch.sum((A * F_all),dim=-1)

        return V


class MG_RAFA(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.backbone = ResNet(last_stride=cfg.last_stride)
        model = resnet50(True)
        state_dict = model.state_dict()
        self.backbone.load_state_dict(state_dict, strict=False)
        self.N = cfg.N
        num_feature = 2048
        D = 16*8
        C = int(num_feature / self.N)
      
        ##multi branch
        self.model = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.branch_fc = nn.ModuleList()
        self.branch_bn=nn.ModuleList()

        ##最终分类
        self.bn=nn.BatchNorm1d(self.N * int(num_feature / self.N))
        self.bn.bias.requires_grad_=False
        self.fc = nn.Linear(self.N * int(num_feature / self.N), num_classes,bias=False)
        self.bn.apply(weights_init_kaiming)
        self.fc.apply(weights_init_classifier)

        self.d = []



        for i in range(self.N):
            d = int(D / pow(2, 2 * i))
            self.d.append(d)
            self.model.append(Model(cfg, C, d))
            self.pool.append(nn.AdaptiveAvgPool2d((int(16 / pow(2, i)), int(8 / pow(2, i)))))
            bn=nn.BatchNorm1d(C)
            bn.requires_grad_=False
            fc=nn.Linear(C, num_classes,bias=False)
            bn.apply(weights_init_kaiming)
            fc.apply(weights_init_classifier)
            self.branch_bn.append(bn)
            self.branch_fc.append(fc)
        for model in self.model:
            model.apply(self._init_parameters)
        self.cls_criterion = CrossEntropyLabelSmoothLoss(num_classes)
        self.metric_criterion = TripletLoss(margin=0.3)

    def _init_parameters(self,m):

        # for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



    def forward(self, x, label=None):
        b, T = x.shape[0], x.shape[1]
        x = x.view(b * T, x.shape[2], x.shape[3], x.shape[4])
        
        x = self.backbone(x)[0]
        B, C, H, W = x.shape
        stride = int(C / self.N)
        final_feature = []
        loss = 0
        for i in range(self.N):
            W_i = x[:, i * stride:(i + 1) * stride, :, :]
            W_i = self.pool[i](W_i)
           
            V = self.model[i](W_i)
            V=self.branch_bn[i](V)
            # print(label.shape)
            if self.training:
               
                logit = self.branch_fc[i](V)
                loss = loss + self.cls_criterion(logit, label) / self.N
                loss = loss + self.metric_criterion(V, label) / self.N
            final_feature.append(V)
        final_features = torch.cat(final_feature, dim=1)
        bn_features=self.bn(final_features)
        if self.training:
         
            logit = self.fc(bn_features)
            loss = loss + self.cls_criterion(logit, label)
            loss = loss + self.metric_criterion(bn_features, label)
            return loss,final_features
        else:
            return bn_features
       



class Config:
        root: str=os.path.join(os.path.abspath('/'),'data','mars')
        S: int = 8
        train_bs:int = 16
        test_bs: int = 72
        weight_decay: float = 5e-4
        max_epochs: int =320
        num_dataloader_wokers: int = 2
        last_stride: int = 1
        save_each_rank: bool = False

        N: int =4
        s: int =1


import logging
import datetime
import time

class Trainer:    
    def __init__(self,rank,world_size,args):
        
        

        cfg=Config()
        self.config=cfg
        # if world_size>1:
        #     distribute=True
        #     os.environ['MASTER_ADDR'] = 'localhost'
        #     os.environ['MASTER_PORT'] =str(args.port)
        #     dist.init_process_group("nccl", rank=rank, world_size=world_size)
        #     torch.cuda.set_device(rank)
        train_loader, num_class = Get_Video_train_DataLoader(os.path.join(cfg.root,'train_path.txt'),os.path.join(cfg.root,'train_info.npy'),
                                                          train_transform,num_workers=cfg.num_dataloader_wokers,S=cfg.S,class_per_batch=cfg.train_bs,distribute=False)
        self.train_loader=train_loader

        val_loader = Get_Video_test_DataLoader(os.path.join(cfg.root,'test_path.txt'), os.path.join(cfg.root,'test_info.npy'),os.path.join(cfg.root,'query_IDX.npy'),
                                           train_transform,batch_size=cfg.test_bs,S=cfg.S,distribute=False)

        self.val_loader=val_loader
        self.backbone=MG_RAFA(cfg,num_class)
        self.backbone=self.backbone.cuda()
        # if world_size>1:
        #     self.backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.optimizer = torch.optim.Adam(self.backbone.parameters(),lr=8e-6, weight_decay=self.config.weight_decay)
        self.rank=rank
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 40,0.5)
        self.backbone=nn.DataParallel(self.backbone)
        # if world_size>1:
        #    self.backbone = torch.nn.parallel.DistributedDataParallel(self.backbone, device_ids=[rank])
        

    def train_one_epoch(self):
        
        for i,(images, labels) in enumerate(self.train_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            self.optimizer.zero_grad(set_to_none=True)
           
            with autocast():
                loss,feature = self.backbone(images,labels)
                loss=torch.mean(loss)
             
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

    #warm up    
    def warmup_lr(self,epoch):
        lr=8e-4* (1/100+epoch/20)

        
        return lr
    

       
    

    def train(self):
        epochs=self.config.max_epochs
        max_map=0
        with tqdm(total=epochs) as pbar:
            for epoch in range(epochs):
                    if epoch<=20:
                        lr=self.warmup_lr(epoch)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr
                    self.backbone.train()
                    self.train_one_epoch()
                    if epoch>20:
                        self.scheduler.step()   
                    if (epoch%5)==0 and epoch!=0:
                            rank1,map=self.do_eval()
                            print('epoch:{}----map:{}--rank1:{}---max-map:{}'.format(epoch,map,rank1,max_map))
                            if map>max_map:
                                max_map=map
                
                                torch.save({'model':self.backbone.state_dict(),'optimizer':self.optimizer.state_dict()},'result.pt')
                    pbar.update(1)
       
        

    def do_eval(self):
        gallery_features = []
        gallery_labels = []
        gallery_cams = []

        self.backbone.eval()
        with torch.no_grad():
            for images, labels, cams in self.val_loader:
                B, C, H, W = images.shape
                images = images.reshape(B // 8, 8, C, H, W)
                images = images.float().cuda()
                feature = self.backbone(images)
                gallery_features.append(feature.detach().cpu())
                gallery_labels.append(labels)
                gallery_cams.append(cams)
        gallery_features = torch.cat(gallery_features, dim=0).numpy()
        gallery_labels = torch.cat(gallery_labels, dim=0).numpy()
        gallery_cams = torch.cat(gallery_cams, dim=0).numpy()
 
        rank1, mAP = Video_Cmc(gallery_features, gallery_labels, gallery_cams, self.val_loader.dataset.query_idx, 10000)
        return rank1[0],mAP
def main(rank,world_size,args):

    
    trainer=Trainer(rank,world_size,args)
  
    trainer.train()



    return trainer
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP, distributed
def main_workers(args):
    world_size=args.gpus
    # if world_size>1:
        
    #     mp.spawn(main, nprocs=world_size, args=(world_size,args))
    # else:
    main(0,world_size,args)

from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


scaler = GradScaler()
import warnings
import random
import argparse
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    arg_parser = argparse.ArgumentParser(description="Train Reid")
    arg_parser.add_argument(
        "--gpus",
        type=int,
        default=1
    ) 
    arg_parser.add_argument(
        "--port",
        type=int,
        default=8000
    )
    args = arg_parser.parse_args()
 
    # torch.backends.cudnn.deterministic=True
    warnings.filterwarnings('ignore')
    main_workers(args)



