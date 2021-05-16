import torch
import torch.nn as nn
from utils.utils import  *


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


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class BNneckHead(nn.Module):
    def __init__(self, in_feat, num_class, pool_layer=nn.Identity()):
        super().__init__()

        self.pool_layer = nn.Sequential(
            pool_layer,
            Flatten()
        )
        self.bnneck = nn.BatchNorm1d(in_feat)
        self.bnneck.bias.reuqires_grad=False
        self.bnneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(in_feat, num_class, bias=False)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feat = self.pool_layer(features)
        bn_feat = self.bnneck(global_feat)
        # evaluation
        if not self.training:
            return bn_feat
        # training
        try:
            pred_class_logits = self.classifier(bn_feat)
        except TypeError:
            pred_class_logits = self.classifier(bn_feat, targets)
        return pred_class_logits, global_feat,bn_feat

