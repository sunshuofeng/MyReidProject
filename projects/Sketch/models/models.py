from models.backbone import *
from models.layers import *
import torch.nn as nn
import math
import torch
import numpy as np
import sys
sys.path.append('../..')
from utils.loss import *
from models.utils import *
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self,cfg,num_class):
        super(Model, self).__init__()

        self.cls_criterion=CrossEntropyLabelSmooth(num_class)
        self.circle_criterion=CircleLoss(m=cfg['Margin'],gamma=80)
        self.N=cfg['N']
        self.M=cfg['M']
        self.lambdak=nn.Parameter(torch.from_numpy(np.array([cfg['lambda_init']]*self.M)))
        self.a0=-1*math.pi
        self.b0=1*math.pi
        self.a1=-1*math.pi/4
        self.b1=-3*math.pi/4
        self.a2=3*math.pi/4
        self.b2=math.pi/4
        self.R_max=cfg['R_max']
        self.backbone_list=nn.ModuleList()
        for i in range(3):
            backbone,feat_dim=build_backbone(cfg['backbone'])
            self.backbone_list.append(backbone)
        self.stride=cfg['stride']
        self.pool_list=nn.ModuleList()
        for i in range(3):
            for j in range(self.stride):
                pool=GeneralizedMeanPoolingP()
                self.pool_list.append(pool)

        self.ASE_FC_list=nn.ModuleList()
        self.ASE_CONV_list=nn.ModuleList()
        for i in range(3):
            for j in range(self.stride):
                fc_block=nn.Sequential(
                    nn.Linear(2048,1024),
                    nn.ReLU(),
                    nn.Linear(1024,2048),
                )
                conv_b=nn.Sequential(
                    nn.Conv2d(2048,1024,kernel_size=1),
                    nn.BatchNorm2d(1024),
                    nn.ReLU()
                )
                self.ASE_FC_list.append(fc_block)
                self.ASE_CONV_list.append(conv_b)

        self.fc_list=nn.ModuleList()
        for i in range(3):
            for j in range(self.stride):
                fc=nn.Linear(1024,num_class)
                self.fc_list.append(fc)

    def ASE(self, x, index):
        feat = self.ASE_FC_list[index](x.view(x.shape[0], -1))
        x = x + torch.mul(x, F.sigmoid(feat).view(x.shape[0], -1, 1, 1))
        x = self.ASE_CONV_list[index](x)
        return x

    def forward(self, x, label):
        theta_list0 = []
        theta_list1 = []
        theta_list2 = []
        for i in range(self.N):
            zi = self.lambdak[:i].sum() / self.lambdak.sum()
            fi = (self.b0 - self.a0) * zi + self.a0
            theta_list0.append(fi)
        for i in range(self.N):
            zi = self.lambdak[:i].sum() / self.lambdak.sum()
            fi = (self.b1 - self.a1) * zi + self.a1
            theta_list1.append(fi)
        for i in range(self.N):
            zi = self.lambdak[:i].sum() / self.lambdak.sum()
            fi = (self.b2 - self.a2) * zi + self.a2
            theta_list2.append(fi)
        theta_lists = np.zeros((3, self.M))
        theta_lists[0] = theta_list0
        theta_lists[1] = theta_list1
        theta_lists[2] = theta_list2
        theta_lists = torch.from_numpy(theta_lists)
        self.grid = torch.zeros((3, 1, self.N, self.M, 2))
        for k in range(3):
            for i in range(self.N):
                for j in range(self.M):
                    self.grid[k, 0, i, j, 0] = self.R_max * j * torch.sin(theta_lists[k, i]) / self.M
                    self.grid[k, 0, i, j, 1] = self.R_max * j * torch.cos(theta_lists[k, i]) / self.M
        self.grid = self.grid.cuda()

        grid_features = []
        for i in range(3):
            x = x.float()
            grid = self.grid[i]
            grid = grid.repeat(x.shape[0], 1, 1, 1)
            grid_feature = F.grid_sample(x, grid)
            grid_features.append(grid_feature)
        loss = 0
        logit_list = []
        final_feature_list = []
        for i in range(3):
            features = self.backbone_list[i](grid_features[i])
            stride = int(features.shape[2] / self.stride)
            for j in range(self.stride):
                feature = features[:, :, j * stride:(j + 1) * stride, :]
                feature = self.pool_list[i * 4 + j](feature)
                feature = self.ASE(feature, i * 4 + j)
                feature = feature.view(feature.shape[0], -1)
                logit = self.fc_list[i * 4 + j](feature)
                logit_list.append(logit)
                final_feature_list.append(feature)
        feat = torch.cat(final_feature_list, dim=1)
        if self.training:
            for logit in logit_list:
                loss = loss + self.cls_criterion(logit, label)
            loss = loss + self.circle_criterion(feat, label)
        else:
            loss = 0
        return feat, loss


