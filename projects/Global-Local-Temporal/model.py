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


class DTP(nn.Module):
    def __init__(self, d):
        super(DTP, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d, out_channels=d, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(in_channels=d, out_channels=d, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(in_channels=d, out_channels=d, kernel_size=3, padding=4, dilation=4)
        self.conv1.apply(weights_init_kaiming)
        self.conv2.apply(weights_init_kaiming)
        self.conv3.apply(weights_init_kaiming)

    ## [bs,d,T]
    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv2(x)
        x2 = self.conv3(x)
        output = []
        output.append(x0)
        output.append(x1)
        output.append(x2)
        output = torch.cat(output, dim=1)
        return output


class TSA(nn.Module):
    def __init__(self, d, a=2):
        super(TSA, self).__init__()
        self.inter_channel = int(d / a)
        self.conv1 = nn.Conv1d(in_channels=d, out_channels=self.inter_channel, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d, out_channels=self.inter_channel, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=d, out_channels=self.inter_channel, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=self.inter_channel, out_channels=d, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(self.inter_channel)
        self.bn2 = nn.BatchNorm1d(self.inter_channel)
        self.relu = nn.ReLU()
        self.conv1.apply(weights_init_kaiming)
        self.conv2.apply(weights_init_kaiming)
        self.conv3.apply(weights_init_kaiming)
        self.conv4.apply(weights_init_kaiming)
        self.bn1.apply(weights_init_kaiming)
        self.bn2.apply(weights_init_kaiming)

    ##bs,3d,T
    def forward(self, x):
        ##bs,3d/2,T
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.relu(self.bn2(self.conv2(x)))
        x2 = self.conv3(x)

        x0 = x0.permute(0, 2, 1)
        f = torch.matmul(x0, x1)
        f = torch.softmax(f, dim=-1)

        M = torch.matmul(x2, f)
        M = self.conv4(M)
        output = M + x
        return output


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
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.dtp = DTP(num_feature)
        self.tsa = TSA(3 * num_feature)

        self.fc = nn.Linear(3 * num_feature, num_classes, bias=False)

        self.fc.apply(weights_init_classifier)

    def forward(self, x):
        batch_size = x.shape[0]
        T = x.shape[1]

        x = x.view(batch_size * T, x.size(2), x.size(3), x.size(4))
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(batch_size, T, -1).permute(0, 2, 1)

        x = self.dtp(x)

        x = self.tsa(x)

        feature = torch.mean(x, dim=-1)
        logit = self.fc(feature)

        out = {}
        out['feature'] = feature
        out['logit'] = logit
        return out





