from torchvision.models import resnet50,resnet18,resnet34,resnet101,resnet152
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch
import torch.nn.functional as F

class Backbone_eff(nn.Module):
    def __init__(self,model):
        super(Backbone_eff, self).__init__()
        self.model=model

    def forward(self,x):
        return self.model.extract_features(x)

def build_backbone(model_name):
    eff_root='efficientnet-'
    if model_name.startswith('b'):
        model_name=eff_root+model_name
        model=EfficientNet.from_pretrained(model_name)
        num_feature=model._bn1.num_features
        backbone=Backbone_eff(model)
    else:
        if model_name=='50':
            model=resnet50(True)
        elif model_name == '18':
            model = resnet18(True)
        elif model_name=='34':
            model=resnet34(True)
        elif model_name=='101':
            model=resne101(True)
        elif model_name=='200':
            model=resne200(True)

        num_feature=model.fc.in_features
        backbone=nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )
    return backbone,num_feature

