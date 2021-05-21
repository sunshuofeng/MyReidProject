from torch.nn import functional as F
from torchvision.models import resnet50
import torch
import torch.nn as nn


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class ICA(nn.Module):
    def __init__(self, num_class, num_feature, bnneck=False):
        super(ICA, self).__init__()
        self.global_pool = nn.AdaptiveMaxPool2d(1)
        self.projection_fc = nn.Linear(num_feature, 512, bias=False)
        self.projection_bn = nn.BatchNorm1d(512)
        self.fc = nn.Linear(512, num_class, bias=False)
        self.projection_fc.apply(weights_init_classifier)
        self.fc.apply(weights_init_classifier)
        self.projection_bn.apply(weights_init_kaiming)
        if bnneck:
            self.projection_bn.bias.requires_grad = False

    def forward(self, x, batch_size, T):
        x = self.global_pool(x)
        x = x.view(batch_size, T, -1)
        feature = torch.mean(x, dim=1)
        feature = self.projection_fc(feature)
        bn_feature = self.projection_bn(feature)
        logit = self.fc(bn_feature)
        return logit, bn_feature, feature

class Baseline(nn.Module):
    def __init__(self, num_classes):
        super(Baseline, self).__init__()
        model = resnet50(True)
        self.backbone = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )
        num_feature = model.fc.in_features
        self.ica = ICA(num_classes, num_feature)

    def forward(self, x):
        batch_size = x.shape[0]
        T = x.shape[1]

        x = x.view(batch_size * T, x.size(2), x.size(3), x.size(4))
        x = self.backbone(x)
        logit,bn_feature,feature=self.ica(x,batch_size,T)
        out = {}
        out['feature'] = bn_feature
        out['logit'] = logit
        return out


