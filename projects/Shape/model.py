
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def _upsample(x):
    h, w = x.size()[2:]
    return F.interpolate(x, size=(h * 2, w * 2), mode='bilinear')


class GBlock(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=F.relu, upsample=False, num_classes=0):
        super(GBlock, self).__init__()

        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch or upsample
        if h_ch is None:
            h_ch = out_ch
        self.num_classes = num_classes

        # Register layrs
        self.c1 = nn.Conv2d(in_ch, h_ch, ksize, 1, pad)
        self.c2 = nn.Conv2d(h_ch, out_ch, ksize, 1, pad)
        if self.num_classes > 0:
            self.b1 = CategoricalConditionalBatchNorm2d(
                num_classes, in_ch)
            self.b2 = CategoricalConditionalBatchNorm2d(
                num_classes, h_ch)
        else:
            self.b1 = nn.BatchNorm2d(in_ch)
            self.b2 = nn.BatchNorm2d(h_ch)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1)

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.tensor, gain=math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.tensor, gain=math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.tensor, gain=1)

    def forward(self, x, y=None, z=None, **kwargs):

        return self.shortcut(x) + self.residual(x, y, z)

    def shortcut(self, x, **kwargs):
        if self.learnable_sc:
            if self.upsample:
                h = _upsample(x)
            h = self.c_sc(h)
            return h
        else:
            return x

    def residual(self, x, y=None, z=None, **kwargs):
        if y is not None:
            h = self.b1(x, y, **kwargs)
        else:
            h = self.b1(x)
        h = self.activation(h)
        if self.upsample:
            h = _upsample(h)
        h = self.c1(h)
        if y is not None:
            h = self.b2(h, y, **kwargs)
        else:
            h = self.b2(h)
        return self.c2(self.activation(h))
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import utils


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolutiReceptive Multi-granularity Representation for
Person Re-Identificationon"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            groups=1,
            base_width=64,
            dilation=1,
            norm_layer=None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = utils.spectral_norm(conv3x3(inplanes, planes, stride))

        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = utils.spectral_norm(conv3x3(planes, planes))

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor):
        identity = x

        out = self.conv1(x)

        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block,
            layers,
            num_classes=1000,
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = utils.spectral_norm(nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                                   bias=False))

        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, blocks,
                    stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                utils.spectral_norm(conv1x1(self.inplanes, planes * block.expansion, stride)),

            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)

        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
        arch,
        block,
        layers,
        pretrained,
        progress,
        **kwargs
):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


class ResNetGenerator(nn.Module):
    """Generator generates 128x128."""

    def __init__(self, num_features=128, dim_z=2048, bottom_width=8, bottom_height=16,
                 activation=F.relu, num_classes=0, distribution='normal'):
        super(ResNetGenerator, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.bottom_height = bottom_height
        self.activation = activation
        self.num_classes = num_classes
        self.distribution = distribution

        self.l1 = nn.Linear(dim_z, 16 * num_features * bottom_width * bottom_height)

        self.block2 = GBlock(num_features * 16, num_features * 16,
                             activation=activation, upsample=True,
                             num_classes=num_classes)
        self.block3 = GBlock(num_features * 16, num_features * 8,
                             activation=activation, upsample=True,
                             num_classes=num_classes)
        self.block4 = GBlock(num_features * 8, num_features * 4,
                             activation=activation, upsample=True,
                             num_classes=num_classes)
        self.block5 = GBlock(num_features * 4, num_features * 2,
                             activation=activation, upsample=True,
                             num_classes=num_classes)
        # self.block6 = GBlock(num_features * 2, num_features*2,
        #                             activation=activation, upsample=True,
        #                             num_classes=num_classes)
        self.b7 = nn.BatchNorm2d(num_features*2)
        self.conv7 = nn.Conv2d(num_features*2, 3, 1, 1)

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.tensor)
        init.xavier_uniform_(self.conv7.weight.tensor)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_height, self.bottom_width)
        for i in [2, 3, 4, 5]:
            h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b7(h))
        return torch.tanh(self.conv7(h))


class DBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            groups=1,
            base_width=64,
            dilation=1,
            norm_layer=None
    ):
        super(DBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = utils.spectral_norm(conv1x1(inplanes, planes))

        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = utils.spectral_norm(conv1x1(planes, planes))

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor):
        identity = x

        out = self.conv1(x)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.relu(out)

        return out


class ImageD(nn.Module):
    def __init__(self):
        super(ImageD, self).__init__()
        self.base = resnet18()
        self.l = utils.spectral_norm(nn.Linear(512, 1))
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.base(x)
        x = torch.sum(self.act(x), dim=(2, 3))
        x = self.l(x)
        return x


class FeatureD(nn.Module):
    def __init__(self):
        super(FeatureD, self).__init__()
        self.block1 = DBasicBlock(2048, 64)
        self.block2 = DBasicBlock(64, 128)
        self.block3 = DBasicBlock(128, 256)
        self.block4 = DBasicBlock(256, 512)
        # self.block5=DBasicBlock(512,1024)
        self.act = nn.LeakyReLU()
        self.l = nn.Linear(512, 1)

    def forward(self, x):
        x = x.view(x.shape[0], -1, 1, 1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # x=self.block5(x)
        x = self.act(x)
        x = x.view(x.shape[0], -1)
        x = self.l(x)
        return x
import torch.nn.functional as F
class Shape_Enocder(nn.Module):
    def __init__(self,cfg,num_class):
        super(Shape_Enocder, self).__init__()
        self.cfg=cfg
        self.flat=Flatten()
        self.backbone,feat_dim=build_backbone(cfg['shape_backbone'])
        if cfg['shape_local']:
            self.has_1_pool = build_pool_conv(feat_dim, 512)
            self.has_2_pool = build_pool_conv(feat_dim, 512)
            self.has_3_pool = build_pool_conv(feat_dim, 512)

            self.has_red = build_reduction_layer(512 * 3, 1024)
            self.head_bn = nn.BatchNorm1d(1024)
            self.head_bn.bias.requires_grad_(False)
            self.head_classfier = nn.Linear(1024, num_class, bias=False)
            self.head_bn.apply(weights_init_kaiming)
            self.head_classfier.apply(weights_init_classifier)

            self.han_pool_1 = GeneralizedMeanPoolingP()
            self.han_c_att_1 = nn.Sequential(nn.Linear(2048, 1024, bias=False), nn.ReLU(),
                                             nn.Linear(1024, 2048, bias=False))
            self.han_c_att_1[0].apply(weights_init_kaiming)
            self.han_c_att_1[1].apply(weights_init_kaiming)
            self.han_pool_2 = GeneralizedMeanPoolingP()
            self.han_c_att_2 = nn.Sequential(nn.Linear(2048, 1024, bias=False), nn.ReLU(),
                                             nn.Linear(1024, 2048, bias=False))
            self.han_c_att_2[0].apply(weights_init_kaiming)
            self.han_c_att_2[1].apply(weights_init_kaiming)
            self.han_pool_3 = GeneralizedMeanPoolingP()
            self.han_c_att_3 = nn.Sequential(nn.Linear(2048, 1024, bias=False), nn.ReLU(),
                                             nn.Linear(1024, 2048, bias=False))
            self.han_c_att_3[0].apply(weights_init_kaiming)
            self.han_c_att_3[1].apply(weights_init_kaiming)
        else:
            self.pool = GeneralizedMeanPoolingP()
            self.bn = nn.BatchNorm1d(feat_dim)
            self.bn.bias.requires_grad_(False)
            self.fc = nn.Linear(feat_dim, num_class,bias=False)
            self.bn.apply(weights_init_kaiming)
            self.fc.apply(weights_init_classifier)

    def HAN_1(self, x):
        c_att = self.han_c_att_1(self.han_pool_1(x).view(x.shape[0], -1))
        c_att = F.sigmoid(c_att).view(x.shape[0], -1, 1, 1)
        feat = x + torch.mul(x, c_att)
        s_att = F.sigmoid(torch.sum(feat, dim=1)).unsqueeze(1)
        han_feat = torch.mul(feat, s_att)
        return han_feat

    def HAN_2(self, x):
        c_att = self.han_c_att_2(self.han_pool_2(x).view(x.shape[0], -1))
        c_att = F.sigmoid(c_att).view(x.shape[0], -1, 1, 1)
        feat = x + torch.mul(x, c_att)
        s_att = F.sigmoid(torch.sum(feat, dim=1)).unsqueeze(1)
        han_feat = torch.mul(feat, s_att)
        return han_feat

    def HAN_3(self, x):
        c_att = self.han_c_att_3(self.han_pool_3(x).view(x.shape[0], -1))
        c_att = F.sigmoid(c_att).view(x.shape[0], -1, 1, 1)
        feat = x + torch.mul(x, c_att)
        s_att = F.sigmoid(torch.sum(feat, dim=1)).unsqueeze(1)
        han_feat = torch.mul(feat, s_att)
        return han_feat

    def forward(self,x):
        x=self.backbone(x)
        if self.cfg['shape_local']:
            height = int(x.shape[2] // 3)
            hf_1 = self.HAN_1(x[:, :, 0 * height:1 * height, :])
            hf_1 = self.has_1_pool(hf_1)
            hf_2 = self.HAN_2(x[:, :, 1 * height:2 * height, :])
            hf_2 = self.has_2_pool(hf_2)
            hf_3 = self.HAN_3(x[:, :, 2 * height:3 * height, :])
            hf_3 = self.has_3_pool(hf_3)
            hf = torch.cat([hf_1, hf_2, hf_3], dim=1)
            hf=hf.reshape(hf.shape[0],-1)
            global_feat = self.has_red(hf)
            feat = self.head_bn(global_feat)
            logit=self.head_classfier(feat)
        else:
            global_feat = self.pool(x)
            global_feat = self.flat(global_feat)
            feat = self.bn(global_feat)
            logit = self.fc(feat)
        if self.training:
            return global_feat,logit
        else:
            return feat,logit

class ColorEncoder(nn.Module):
    def __init__(self,cfg):
        super(ColorEncoder,self).__init__()
        self.backbone,feat_dim=build_backbone(cfg['color_backbone'])
        self.pool=GeneralizedMeanPoolingP()
        self.bn=nn.BatchNorm1d(feat_dim)
        self.bn.bias.requires_grad_(False)
        self.bn.apply(weights_init_kaiming)

    def forward(self,x):
        x=self.backbone(x)
        global_feat=self.pool(x)
        global_feat=global_feat.view(global_feat.shape[0],-1)
        feat=self.bn(global_feat)
        if self.training:
            return global_feat
        else:
            return feat


from tqdm import tqdm
class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.ES = Shape_Enocder(cfg, num_class)
        self.EC = ColorEncoder(cfg)
        self.DF = FeatureD()
        self.G = ResNetGenerator(64)
        self.DI = ImageD()
        self.num_class = num_class
        self.cfg = cfg
        self.DF.apply(weights_init_kaiming)
        self.DI.apply(weights_init_kaiming)
        self.G.apply(weights_init_kaiming)


