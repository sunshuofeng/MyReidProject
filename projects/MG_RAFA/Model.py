import torch
import torch.nn as nn


class CrossEntropyLabelSmoothLoss0(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmoothLoss0, self).__init__()
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
        targets = labels.scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = (xx + yy).float()
    dist.addmm_(1, -2, x.float(), y.float().t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=True):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)





class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride





    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, cfg ,last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=last_stride)




    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.relu(x)    # add missed relu
        x = self.maxpool(x)
        x= self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

from torchvision.models import resnet50

##没有初始化，各位可以先初始化里面的各种参数再训练
class Model(nn.Module):
    def __init__(self, cfg, num_feature, D):
        super(Model, self).__init__()
        self.relu = nn.ReLU()

        ##论文使用的是1*1卷积，实质上和全连接层一模一样，因为本质上输入是二维的,K个特征节点，每个节点是C维特征向量
        self.W_u = nn.Linear(num_feature, int(num_feature / cfg['s']))
        self.bn_u = nn.BatchNorm1d(int(num_feature / cfg['s']))

        self.W_v = nn.Linear(num_feature, int(num_feature / cfg['s']))
        self.bn_v = nn.BatchNorm1d(int(num_feature / cfg['s']))

        self.W_phi = nn.Linear(num_feature, int(num_feature / cfg['s']))
        self.bn_phi = nn.BatchNorm1d(int(num_feature / cfg['s']))

        self.W_psi = nn.Linear(D, int(D / cfg['s']))
        self.bn_psi = nn.BatchNorm1d(int(D / cfg['s']))

        self.W_theta = nn.Linear(int(D / cfg['s']) + int(num_feature / cfg['s']), num_feature)
        self.bn_theta = nn.BatchNorm1d(num_feature)

    ##这里我假设T=8
    def forward(self, x):
        B, C, H, W = x.shape
        b = int(B / 8)
        T = 8

        ##普通特征节点
        x0 = x.view(B, C, -1)
        x0 = x0.permute(0, 2, 1).contiguous()
        F_all = x0.view(B * H * W, C)

        ##平均特征节点
        F_R = x0.view(b, T, H * W, C)
        F_R = torch.mean(F_R, dim=1)
        F_R = F_R.view(b * H * W, C)

        F_all_u = self.relu(self.bn_u(self.W_u(F_all)))
        F_R_v = self.relu(self.bn_v(self.W_v(F_R)))

        F_all_u = F_all_u.view(b, T, H * W, -1)
        F_R_v = F_R_v.view(b, 1, H * W, -1)

        ##点乘操作
        Relation = torch.matmul(F_all_u, F_R_v.permute(0, 1, 3, 2))
        Relation = Relation.view(B * H * W, -1)

        x1 = self.relu(self.bn_phi(self.W_phi(F_all)))
        Relation = self.relu(self.bn_psi(self.W_psi(Relation)))

        A = self.relu(self.bn_theta(self.W_theta(torch.cat([x1, Relation], dim=1))))
        A = nn.Softmax(dim=-1)(A)
        V = torch.sum((A * F_all).view(b, T * H * W, -1), dim=1)
        return V


class MG_RAFA(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.backbone = ResNet(1)
        model = resnet50(True)
        state_dict = model.state_dict()
        self.backbone.load_state_dict(state_dict, strict=False)
        self.N = cfg['N']
        num_feature = 2048
        D = 16 * 8
        self.model = nn.ModuleList()
        self.pool = nn.ModuleList()
        C = int(num_feature / self.N)
        self.branch_fc = nn.ModuleList()
        self.fc = nn.Linear(self.N * int(num_feature / self.N), num_classes)
        self.d = []
        for i in range(self.N):
            d = int(D / pow(2, 2 * i))
            self.d.append(d)
            self.model.append(Model(cfg, C, d))
            self.pool.append(nn.AdaptiveAvgPool2d((int(16 / pow(2, i)), int(8 / pow(2, i)))))
            self.branch_fc.append(nn.Linear(C, num_classes))

        self.cls_criterion = CrossEntropyLabelSmoothLoss0(num_classes)
        self.metric_criterion = TripletLoss(margin=cfg['margin'])

    def forward(self, x, label=None):
        b, T = x.shape[0], x.shape[1]
        x = x.view(b * T, x.shape[2], x.shape[3], x.shape[4])
        x = self.backbone(x)
        B, C, H, W = x.shape
        stride = int(C / self.N)
        final_feature = []
        loss = 0
        for i in range(self.N):
            W_i = x[:, i * stride:(i + 1) * stride, :, :]
            W_i = self.pool[i](W_i)

            V = self.model[i](W_i)
            if self.training:
                logit = self.branch_fc[i](V)
                loss = loss + self.cls_criterion(logit, label) / self.N
                loss = loss + self.metric_criterion(V, label) / self.N
            final_feature.append(V)
        final_features = torch.cat(final_feature, dim=1)

        if self.training:
            logit = self.fc(final_features)
            loss = loss + self.cls_criterion(logit, label)
            loss = loss + self.metric_criterion(final_features, label)
        out = {}
        out['loss'] = loss
        out['feature'] = final_features

        return out
