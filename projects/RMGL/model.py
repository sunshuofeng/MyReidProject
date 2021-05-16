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


class ABP(nn.Module):
    def __init__(self, ns):
        super(ABP, self).__init__()
        self.ns = ns

    def make_I(self, x):
        height, width = x.shape[2], x.shape[3]
        a = x.view(x.shape[0], x.shape[1], -1)
        value, index = torch.max(a, 2)
        new_value = value.unsqueeze(-1).repeat(1, 1, height * width)
        mask = a == new_value
        I = torch.zeros_like(a)
        I[mask] = 1
        I=I.view(x.shape[0],x.shape[1],height,width)
        return I

    def make_H(self, x):
        I=self.make_I(x)
        height = x.shape[2]
        batch = x.shape[0]
        H = torch.zeros([batch, height]).to(x.device)
        for i in range(height):
            H[:, i] = I[:, :, :i, :].sum(dim=(1, 2, 3))
        return H

    def make_hk(self, x):
        H = self.make_H(x)
        C = x.shape[1]
        hks = torch.zeros(H.shape[0], self.ns + 1)
        hks[:, self.ns] = x.shape[2]
        
       
        for i in range(H.shape[0]):
            k = 1
            for j in range(1,H.shape[1]):
                if k == self.ns:
                    break
                if H[i, j] <= int(k * C / self.ns) and H[i, j + 1] > int(k * C / self.ns):
                 
                    hks[i, k] = j
                    k += 1
        return hks

    def forward(self, x):
#         print(x.shape)
        hk = self.make_hk(x)
#         print(hk)
        F = x.sum(dim=(2, 3)) / x.shape[-1]
        hk_sub = torch.zeros(x.shape[0], self.ns, 1)
        for i in range(1, self.ns + 1):
            hk_sub[:, i - 1, 0] = hk[:, i] - hk[:, i - 1]
        hk_sub = hk_sub.to(x.device)
        F = F.unsqueeze(1)
        F = F.repeat(1, self.ns, 1)
        F = F / hk_sub
       
        F = F.view(F.shape[0], -1)
        return F


class Local_n_branch(nn.Module):
    def __init__(self, backbone, n, num_class):
        super(Local_n_branch, self).__init__()
        self.n = n
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.poola = nn.AdaptiveAvgPool2d(1)
        self.poolb = ABP(n)

        self.reduction_linear = nn.Linear(2048 * (n + 1), 256 * (n + 1))
        self.dropout = nn.Dropout()
        self.bn = nn.BatchNorm1d(256 * (n + 1))
        self.bn.bias.requires_grad_(False)
        self.fc = nn.Linear(256 * (n + 1), num_class, bias=False)
        self.bn.apply(weights_init_kaiming)
        self.fc.apply(weights_init_classifier)

    def path_a_forward(self, x):
        gf = self.layer2(x)
        gf = self.layer3(gf)
        gf = self.layer4(gf)
        gf = self.poola(gf)
        return gf

    def path_b_forward(self, model, x):
        height = x.shape[2]

        x0 = x[:, :, :int(height / 2), :]
        x1 = x[:, :, int(height / 2):, :]
        x0 = model(x0)
        x1 = model(x1)
        x = torch.cat([x0, x1], dim=2)
        return x

    def forward(self, x):
        gf = self.path_a_forward(x)
        gf = gf.view(gf.shape[0], -1)

        batch = x.shape[0]
        height = x.shape[2]
        inputs = []
        stride = int(height / self.n)
        for i in range(self.n):
            inputs.append(x[:, :, i * stride:(i + 1) * stride, :])
        lf = torch.cat(inputs, dim=0).cuda()
        lf = self.path_b_forward(self.layer2, lf)
        lf = self.path_b_forward(self.layer3, lf)
        lf = self.path_b_forward(self.layer4, lf)
        outputs = []
        for i in range(self.n):
            outputs.append(lf[i * batch:(i + 1) * batch])
        lf = torch.cat(outputs, dim=2).cuda()
        
        lf = self.poolb(lf)
        
        
        feature = torch.cat([gf, lf], dim=1)
        global_feature = self.reduction_linear(feature)
        feature = self.bn(global_feature)
        logit = self.fc(feature)
        return global_feature, feature, logit

from torchvision.models import resnet50
class RMGLModel(nn.Module):
    def __init__(self, num_class):
        super(RMGLModel, self).__init__()
        backbone = resnet50(True)
        self.share_backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1
        )
        self.local_2_branch = Local_n_branch(backbone, 2, num_class)
        self.local_3_branch = Local_n_branch(backbone, 3, num_class)
        self.cls_criterion = CrossEntropyLabelSmoothLoss(num_class)
        self.circle_criterion = CircleLoss(m=0.6, gamma=80)

    def forward(self, x, labels):
        x = self.share_backbone(x)

        global_feat0, feat0, logit0 = self.local_2_branch(x)
        global_feat1, feat1, logit1 = self.local_3_branch(x)
        loss = 0
        if self.training:
            loss = loss + self.cls_criterion(logit0, labels)
            
            loss = loss + self.cls_criterion(logit1, labels)
            
            loss = loss + self.circle_criterion(global_feat0, labels)
          
            loss = loss + self.circle_criterion(global_feat1, labels)
           
        feat = torch.cat([feat0, feat1], dim=1)
        out={}
        out['final_loss']=loss
        out['feature']=feat
        return out
