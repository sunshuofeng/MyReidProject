import torch
import torch.nn as nn
class CrossEntropyLabelSmoothLoss(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmoothLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """

        log_probs = self.logsoftmax(inputs)

        labels = torch.zeros(log_probs.size()).to(targets.device)
        targets=labels.scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


from typing import Tuple
from torch import Tensor
def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


def normalize(x, axis=-1):

	x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
	return x


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()


    def forward(self,feat,label) -> Tensor:

        feat=normalize(feat,axis=-1)
        sp, sn = convert_label_to_similarity(feat, label)
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


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
class SNR(nn.Module):
    def __init__(self, inchannel):
        super(SNR, self).__init__()
        self.In = nn.InstanceNorm2d(inchannel)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.w1 = nn.Linear(inchannel, int(inchannel / 16))
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(int(inchannel / 16), inchannel)

    def get_ak(self, R):
        x = self.pool(R)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.w1(x))
        x = nn.Sigmoid()(self.w2(x))
        return x

    def compute_dist(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = (xx + yy).float()
        dist.addmm_(1, -2, x.float(), y.float().t())
        dist = dist.clamp(min=1e-12).sqrt()
        return dist

    def compute_loss(self, dista, distb, labels):
        loss = 0
        N = dista.size(0)
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
        dista_ap = dista[is_pos]
        dista_an = dista[is_neg]
        distb_ap = distb[is_pos]
        distb_an = distb[is_neg]
        loss = loss + nn.Softplus()(dista_ap - distb_ap).mean()

        loss = loss + nn.Softplus()(distb_an - dista_an).mean()
        return loss

    def forward(self, x, labels):
        loss = 0
        if self.training:
            x_p, x_n, x_, F_x = self.__forward(x)
            dist_p = self.compute_dist(x_p, x_p)
            dist_ = self.compute_dist(x_, x_)
            dist_n = self.compute_dist(x_n, x_n)
            loss = loss + self.compute_loss(dist_p, dist_, labels)
            loss = loss + self.compute_loss(dist_, dist_n, labels)
        else:
            F_x = self.__forward(x)
        return F_x, loss

    def __forward(self, f):
        f_ = self.In(f)
        R = f - f_
        ak = self.get_ak(R)
        ak = ak.view(ak.shape[0], ak.shape[1], 1, 1)
        R_p = ak * R
        F_p = R_p + f_
        f_p = self.pool(F_p)
        if self.training:

            R_n = (1 - ak) * R
            F_n = R_n + f_

            f_n = self.pool(F_n)
            f_ = self.pool(f_)
            return f_p.squeeze(), f_n.squeeze(), f_.squeeze(), F_p
        else:
            return F_p

from torchvision.models import resnet50
class SNRModel(nn.Module):
    def __init__(self, num_class):
        super(SNRModel, self).__init__()
        backbone = resnet50(True)
        stride = 1
        backbone.layer4[0].downsample[0].stride = stride
        backbone.layer4[0].conv2.stride = stride
        self.conv_start = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.snr1 = SNR(256)
        self.layer2 = backbone.layer2
        self.snr2 = SNR(512)
        self.layer3 = backbone.layer3
        self.snr3 = SNR(1024)
        self.layer4 = backbone.layer4
        self.snr4 = SNR(2048)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(2048)
        self.bn.bias.requires_grad_(False)
        self.fc = nn.Linear(2048, num_class, bias=False)
        self.bn.apply(weights_init_kaiming)
        self.fc.apply(weights_init_classifier)
        self.cls_criterion = CrossEntropyLabelSmoothLoss(num_class)
        self.circle_criterion = CircleLoss(m=1.0, gamma=80)

    def forward(self, x, labels):
        loss = 0
        x = self.conv_start(x)
        x = self.layer1(x)
        x, snr_loss = self.snr1(x, labels)
        loss = loss + 0.1 * snr_loss
        x = self.layer2(x)
        x, snr_loss = self.snr2(x, labels)
        loss = loss + 0.1 * snr_loss
        x = self.layer3(x)
        x, snr_loss = self.snr3(x, labels)
        loss = loss + 0.5 * snr_loss
        x = self.layer4(x)
        x, snr_loss = self.snr4(x, labels)
        loss = loss + 0.5 * snr_loss
        global_feat = self.pool(x)
        global_feat = global_feat.view(x.shape[0], -1)
        feat = self.bn(global_feat)
        logit = self.fc(feat)
        if self.training:
            loss = loss + self.cls_criterion(logit, labels)
            loss = loss + self.circle_criterion(feat, labels)
        out = {}
        out['final_loss'] = loss
        out['feature'] = feat
        return out

