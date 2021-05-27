class AFA(nn.Module):
    def __init__(self, std=0.025):
        super(AFA, self).__init__()
        self.std = std

    ##输入维度：【b,T,c】
    def forward(self, x):
        xc = torch.mean(x, dim=1, keepdim=True)
        xd = x - xc
        noise = 1
        if self.training:
            noise = torch.normal(mean=1, std=self.std, size=[x.shape[0], x.shape[1]]).cuda()
            noise = noise.unsqueeze(-1)
        #         print(xd.shape)
        #         print(noise.shape)

        new_x = xc + noise * xd
        return new_x, xc.squeeze(), xd


class Baseline(nn.Module):
    def __init__(self, num_classes):
        super(Baseline, self).__init__()
        model = resnet50(True)
        model.layer4[0].downsample[0].stride = 1
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
        self.bn=nn.BatchNorm1d(num_feature)
        self.bn.apply(weights_init_kaiming)
        self.afa = AFA()

    #         self.ica=ICA(num_classes,num_feature)

    def forward(self, x):

        batch_size = x.shape[0]
        T = x.shape[1]

        x = x.view(batch_size * T, x.size(2), x.size(3), x.size(4))
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(batch_size, T, -1)
        new_x, xc, xd = self.afa(x)
        new_x=new_x.view(batch_size*T,-1)
        new_x=self.bn(new_x)
        logit = self.fc(new_x)
        logit=logit.view(batch_size,T,-1)
        logit=torch.softmax(logit,dim=-1)
        logit=torch.mean(logit,dim=1)

        out = {}
        out['logit'] = logit
        #         out['feature']=new_x

        out['feature'] = xc

        out['xd'] = xd

        return out






