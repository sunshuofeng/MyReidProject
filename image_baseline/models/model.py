from models.layers import *
from models.backbone import *
import torch
import torch.nn as nn
from torchvision.models import resnet50
class Model(nn.Module):
    def __init__(self,cfg,num_class):
        super(Model, self).__init__()
        self.backbone = ResNet(cfg, 1)
        model = resnet50(True)
        state_dict = model.state_dict()
        self.backbone.load_state_dict(state_dict, strict=False)
        pool_layer=nn.AdaptiveAvgPool2d(1)
        self.neck=BNneckHead(2048,num_class,pool_layer=pool_layer)

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



