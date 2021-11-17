import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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


#         if m.bias:
#             nn.init.constant_(m.bias, 0.0)



class CenterLoss(nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

            

    def forward(self, x,labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = torch.pow(x.float(), 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers.float(), 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x.float(), self.centers.float().t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        #dist = []
        #for i in range(batch_size):
        #    value = distmat[i][mask[i]]
        #    value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
        #    dist.append(value)
        #dist = torch.cat(dist)
        #loss = dist.mean()
        return loss

        
    
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




class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, distance='cosine'):
        super(TripletLoss, self).__init__()
        if distance not in ['euclidean', 'cosine']:
            raise KeyError("Unsupported distance: {}".format(distance))
        self.distance = distance
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        if self.distance == 'euclidean':
            dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
            dist = dist + dist.t()
            dist.addmm_(1, -2, inputs, inputs.t())
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        elif self.distance == 'cosine':
            fnorm = torch.norm(inputs, p=2, dim=1, keepdim=True)
            l2norm = inputs.div(fnorm.expand_as(inputs))
            dist = - torch.mm(l2norm, l2norm.t())

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss






def cdist(feat1, feat2):
    """Cosine distance"""
    feat1 = torch.FloatTensor(feat1)  # .cuda()
    feat2 = torch.FloatTensor(feat2)  # .cuda()
    feat1 = torch.nn.functional.normalize(feat1, dim=1)
    feat2 = torch.nn.functional.normalize(feat2, dim=1).transpose(0, 1)
    dist = -1 * torch.mm(feat1, feat2)
    return dist.cpu().numpy()


def np_cdist(feat1, feat2):
    """Cosine distance"""
    feat1_u = feat1 / np.linalg.norm(feat1, axis=1, keepdims=True)  # n * d -> n
    feat2_u = feat2 / np.linalg.norm(feat2, axis=1, keepdims=True)  # n * d -> n
    return -1 * np.dot(feat1_u, feat2_u.T)


def np_norm_eudist(feat1, feat2):
    feat1_u = feat1 / np.linalg.norm(feat1, axis=1, keepdims=True)  # n * d -> n
    feat2_u = feat2 / np.linalg.norm(feat2, axis=1, keepdims=True)  # n * d -> n
    feat1_sq = np.sum(feat1_M * feat1, axis=1)
    feat2_sq = np.sum(feat2_M * feat2, axis=1)
    return np.sqrt(feat1_sq.reshape(-1, 1) + feat2_sq.reshape(1, -1) - 2 * np.dot(feat1_M, feat2.T) + 1e-12)


def sqdist(feat1, feat2, M=None):
    """Mahanalobis/Euclidean distance"""
    if M is None: M = np.eye(feat1.shape[1])
    feat1_M = np.dot(feat1, M)
    feat2_M = np.dot(feat2, M)
    feat1_sq = np.sum(feat1_M * feat1, axis=1)
    feat2_sq = np.sum(feat2_M * feat2, axis=1)
    return feat1_sq.reshape(-1, 1) + feat2_sq.reshape(1, -1) - 2 * np.dot(feat1_M, feat2.T)

def Video_Cmc(features, ids, cams, query_idx, rank_size):
    """
    features: numpy array of shape (n, d)
    label`s: numpy array of shape (n)
    """
    # Sample query

    data = {'feature': features, 'id': ids, 'cam': cams}
    q_idx = query_idx
    g_idx = np.arange(len(ids))
    q_data = {k: v[q_idx] for k, v in data.items()}
    g_data = {k: v[g_idx] for k, v in data.items()}
    if len(g_idx) < rank_size: rank_size = len(g_idx)

    CMC, mAP = Cmc(q_data, g_data, rank_size)

    return CMC, mAP


def Cmc(q_data, g_data, rank_size):
    n_query = q_data['feature'].shape[0]
    n_gallery = g_data['feature'].shape[0]

    dist = np_cdist(q_data['feature'], g_data['feature'])  # Reture a n_query*n_gallery array

    cmc = np.zeros((n_query, rank_size))
    ap = np.zeros(n_query)

    for k in range(n_query):
        good_idx = np.where((q_data['id'][k] == g_data['id']) & (q_data['cam'][k] != g_data['cam']))[0]
 
        junk_mask1 = (g_data['id'] == -1)
        junk_mask2 = (q_data['id'][k] == g_data['id']) & (q_data['cam'][k] == g_data['cam'])
        junk_idx = np.where(junk_mask1 | junk_mask2)[0]
        score = dist[k, :]
        sort_idx = np.argsort(score)
        sort_idx = sort_idx[:rank_size]

        ap[k], cmc[k, :] = Compute_AP(good_idx, junk_idx, sort_idx)

    CMC = np.mean(cmc, axis=0)
    mAP = np.mean(ap)
    return CMC, mAP


def Compute_AP(good_image, junk_image, index):
    cmc = np.zeros((len(index),))
    ngood = len(good_image)

    old_recall = 0
    old_precision = 1.
    ap = 0
    intersect_size = 0
    j = 0
    good_now = 0
    njunk = 0
    for n in range(len(index)):
        flag = 0
        if np.any(good_image == index[n]):
            cmc[n - njunk:] = 1
            flag = 1  # good image
            good_now += 1
        if np.any(junk_image == index[n]):
            njunk += 1
            continue  # junk image

        if flag == 1:
            intersect_size += 1
        recall = intersect_size / ngood
        precision = intersect_size / (j + 1)
        ap += (recall - old_recall) * (old_precision + precision) / 2
        old_recall = recall
        old_precision = precision
        j += 1

        if good_now == ngood:
            return ap, cmc
    return ap, cmc



IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_txt(root):
    train_imgs = []

    data_dir = os.path.join(root, 'bbox_train')
    # 获取所有的id
    ids = sorted(os.listdir(data_dir))
    for id in ids:
        # 获取id下的所有图片
        images = sorted(os.listdir(os.path.join(data_dir, id)))
        for image in images:
            if is_image_file(image):
                # 添加图片的路径
                train_imgs.append(os.path.abspath(os.path.join(data_dir, id, image)))
    train_imgs = np.array(train_imgs)
    # 存到txt中
    np.savetxt('train_path.txt', train_imgs, fmt='%s', delimiter='\n')

    # Test与上面同理
    test_imgs = []
    data_dir = os.path.join(root, 'bbox_test')
    ids = sorted(os.listdir(data_dir))
    for id in ids:
        images = sorted(os.listdir(os.path.join(data_dir, id)))
        for image in images:
            if is_image_file(image):
                test_imgs.append(os.path.abspath(os.path.join(data_dir, id, image)))
    test_imgs = np.array(test_imgs)
    np.savetxt('test_path.txt', test_imgs, fmt='%s', delimiter='\n')
def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

import random
from torchvision.models import resnet50


T=6

class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)

  



