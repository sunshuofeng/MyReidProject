import torch
import torch.nn as nn


class OmniSoftMax(nn.Module):
    def __init__(self, num_features, num_classes, cls_type,
                 with_queue=False, l2_norm=False, scalar=1.0, momentum=0.5):
        super(OmniSoftMax, self).__init__()
        self.identify_mode = cls_type  # (sfm, oim)
        assert cls_type in ("sfm", "oim")
        self.num_features = num_features
        self.num_classes = num_classes
        self.momentum = momentum
        self.with_queue = with_queue
        self.l2_norm = l2_norm
        if "requires_grad" == scalar:  # requires_grad
            self.scalar = nn.Parameter(torch.tensor(1, dtype=torch.float).fill_(10.0), requires_grad=True)
        else:
            self.scalar = scalar

        if "sfm" == self.identify_mode:
            self.weight = nn.Parameter(torch.Tensor(num_classes, num_features), requires_grad=True)
            nn.init.normal_(self.weight, std=0.001)
        elif "oim" == self.identify_mode:
            assert self.l2_norm
            self.register_buffer('weight', torch.zeros(num_classes, num_features, requires_grad=False))
        else:
            raise NotImplementedError

        self.register_buffer('zero_loss', torch.zeros(1, requires_grad=False))
        if with_queue:
            self.register_buffer('unlabeled_input_count', torch.zeros(1, dtype=torch.int32, requires_grad=False))
            self.num_unlabeled = 5000 if num_classes > 1000 else 500
            self.register_buffer('unlabeled_queue',
                                 torch.zeros(self.num_unlabeled, num_features, requires_grad=False))

    def forward(self, inputs, targets):
        weight = self.weight
        if self.l2_norm:
            inputs = F.normalize(inputs, dim=1)  # TODO(bug) detach denominator causes NaN
            weight = F.normalize(weight, dim=1)

        if not self.with_queue:
            predicts = inputs.mm(weight.clone().t()) if "oim" == self.identify_mode else inputs.mm(weight.t())
            if "oim" == self.identify_mode:
                # Update
                for x, y in zip(inputs, targets):
                    self.weight[y] = self.momentum * self.weight[y] + (1. - self.momentum) * x.detach().clone()
                    self.weight[y] = F.normalize(self.weight[y], dim=0)
            output_targets = targets
        else:
            labeled_inputs = inputs[targets > -1]
            labeled_targets = targets[targets > -1]
            unlabeled_inputs = inputs[targets == -1]

            # Counts valid unlabeled input
            self.unlabeled_input_count += unlabeled_inputs.size(0)
            self.unlabeled_input_count.fill_(min(self.num_unlabeled, self.unlabeled_input_count.item()))
            # Update the unlabeled queue before calculating loss, so that bbox features inside the same
            # image can compete with each other. These features are already l2-normalized
            self.unlabeled_queue = torch.cat([self.unlabeled_queue, unlabeled_inputs.detach().clone()])[
                                   -self.num_unlabeled:]
            if (targets > -1).sum().item() == 0:
                return None, None
            valid_unlabeled_queue = self.unlabeled_queue[-self.unlabeled_input_count.item():] \
                if self.unlabeled_input_count > 0 else self.unlabeled_queue[0:0]

            predicts = labeled_inputs.mm(torch.cat([weight.clone(), valid_unlabeled_queue]).t()) \
                if "oim" == self.identify_mode else \
                labeled_inputs.mm(torch.cat([weight, valid_unlabeled_queue]).t())
            # Update
            if "oim" == self.identify_mode:
                self.weight[labeled_targets] = self.momentum * self.weight[labeled_targets] + \
                                               (1. - self.momentum) * labeled_inputs.detach().clone()
                self.weight[labeled_targets] = F.normalize(self.weight[labeled_targets], p=2, dim=1)
            output_targets = labeled_targets

        predicts = predicts * self.scalar if self.l2_norm else predicts
        return predicts, output_targets




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
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
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


class Criterion(nn.Module):
    def __init__(self,cfg,num_class):
        super(Criterion, self).__init__()
        self.cls_criterion=CrossEntropyLabelSmoothLoss(num_class)
        if cfg['metric_learning']=='Triplet':
            self.metric_criterion=TripletLoss(cfg['margin'])
        else:
            self.metric_criterion=CircleLoss(m=cfg['margin'])

    def forward(self,out,label):
        loss=0
        loss=loss+self.cls_criterion(out['logit'],label)
        loss=loss+self.metric_criterion(out['global_feature'],label)
        return loss