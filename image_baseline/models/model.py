from models.layers import *
from models.backbone import *
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self,cfg,num_class):
        super(Model, self).__init__()
        self.backbone,self.num_feature=build_backbone(cfg['backbone'])
        if cfg['pool_layer']=='AVG':
            pool_layer=nn.AdaptiveAvgPool2d(1)
        elif cfg['pool_layer']=='MAX':
            pool_layer=nn.AdaptiveMaxPool2d(1)
        elif cfg['pool_layer']=='GEM':
            pool_layer=GeneralizedMeanPoolingP()
        else:
            pool_layer=nn.Identity()
        self.neck=BNneckHead(self.num_feature,num_class,pool_layer=pool_layer)

    def forward(self,x):
        x=self.backbone(x)
        out={}
        if self.training:
            logit,global_feature,bn_feature=self.neck(x)
            out['logit']=logit
            out['global_feature']=global_feature
            out['feature']=bn_feature
        else:
            feature=self.neck(x)
            out['feature']=feature
        return out



