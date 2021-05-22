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
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_feature, num_classes, bias=False)
        self.fc.apply(weights_init_classifier)

    def forward(self, x):
        batch_size = x.shape[0]
        T = x.shape[1]

        x = x.view(batch_size * T, x.size(2), x.size(3), x.size(4))
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(batch_size, T, -1)
        feature = torch.mean(x, dim=1)
        logit = self.fc(feature)
        out = {}
        out['feature'] = feature
        out['logit'] = logit
        return out


nn.BatchNorm1d().bias.requires_grad

