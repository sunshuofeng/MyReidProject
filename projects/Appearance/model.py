
##AP3D核心就是这个下面三个函数，因为整个模型本质上是resnet50，问题 就是如何用3d卷积去替换2d卷积
##这篇论文的对齐模块并不是应用再每个3d卷积前的，有的就是普通的3d卷积，详情看第二个函数C2D，本质上就是一个2d卷积，只不过是以3d的形式存在(1,kernel_size,kernel_size)，用这个C2D去替换resnet50原来的2d卷积
##当然有些3d卷积前会用到对齐模块，用了对齐模块的3d卷积会与上面的C2D不同

##对齐模块，对齐就是非常简单求点与点之间的相似度，
class APM(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=3, temperature=4, contrastive_att=True):
        super(APM, self).__init__()

        self.time_dim = time_dim 
        self.temperature = temperature
        self.contrastive_att = contrastive_att

        padding = (0, 0, 0, 0, (time_dim-1)//2, (time_dim-1)//2)
        self.padding = nn.ConstantPad3d(padding, value=0)

        self.semantic_mapping = nn.Conv3d(in_channels, out_channels, \
                                          kernel_size=1, bias=False)          
        if self.contrastive_att:  
            self.x_mapping = nn.Conv3d(in_channels, out_channels, \
                                       kernel_size=1, bias=False)
            self.n_mapping = nn.Conv3d(in_channels, out_channels, \
                                       kernel_size=1, bias=False)
            self.contrastive_att_net = nn.Sequential(nn.Conv3d(out_channels, 1, \
                                kernel_size=1, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, t, h, w = x.size()
        N = self.time_dim
        
        #3因为与邻居进行对齐，所以找到邻居的索引
        neighbor_time_index = torch.cat([(torch.arange(0,t)+i).unsqueeze(0) for i in range(N) if i!=N//2], dim=0).t().flatten().long()

        ##进行降维，获取特征图
        semantic = self.semantic_mapping(x) # (b, c/16, t, h, w)
        x_norm = F.normalize(semantic, p=2, dim=1) # (b, c/16, t, h, w)
        ##这里的padding是因为每个人有两个邻居，整体而言视频序列会多出两个帧
        x_norm_padding = self.padding(x_norm) # (b, c/16, t+2, h, w)
        x_norm_expand = x_norm.unsqueeze(3).expand(-1, -1, -1, N-1, -1, -1).permute(0, 2, 3, 4, 5, 1).contiguous().view(-1, h*w, c//16) # (b*t*2, h*w, c/16) 
        neighbor_norm = x_norm_padding[:, :, neighbor_time_index, :, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, c//16, h*w) # (b*t*2, c/16, h*w) 
        
        ##然后进行矩阵乘法求点与点余弦相似度
        similarity = torch.matmul(x_norm_expand, neighbor_norm) * self.temperature # (b*t*2, h*w, h*w)
        similarity = F.softmax(similarity, dim=-1) # (b*t*2, h*w, h*w)
        


        x_padding = self.padding(x)

        neighbor = x_padding[:, :, neighbor_time_index, :, :].permute(0, 2, 3, 4, 1).contiguous().view(-1, h*w, c)

        ##根据点的相似度进行特征聚合
        neighbor_new = torch.matmul(similarity, neighbor).view(b, t*(N-1), h, w, c).permute(0, 4, 1, 2, 3) # (b, c, t*2, h, w)

        # contrastive attention

        ##论文里提到的如果不是同一个地方就用attention去掉
        if self.contrastive_att:
            x_att = self.x_mapping(x.unsqueeze(3).expand(-1, -1, -1, N-1, -1, -1).contiguous().view(b, c, (N-1)*t, h, w).detach())
            n_att = self.n_mapping(neighbor_new.detach())
            contrastive_att = self.contrastive_att_net(x_att * n_att)    
            neighbor_new = neighbor_new * contrastive_att

        # integrating feature maps
        x_offset = torch.zeros([b, c, N*t, h, w], dtype=x.data.dtype, device=x.device.type)
        x_index = torch.tensor([i for i in range(t*N) if i%N==N//2])
        neighbor_index = torch.tensor([i for i in range(t*N) if i%N!=N//2])
        x_offset[:, :, x_index, :, :] += x
        x_offset[:, :, neighbor_index, :, :] += neighbor_new

        return x_offset

##普通的用3d卷积去替换2d卷积
class C2D(nn.Module):
    def __init__(self, conv2d, **kwargs):
        super(C2D, self).__init__()

        # conv3d kernel
        kernel_dim = (1, conv2d.kernel_size[0], conv2d.kernel_size[1])
        stride = (1, conv2d.stride[0], conv2d.stride[0])
        padding = (0, conv2d.padding[0], conv2d.padding[1])
        self.conv3d = nn.Conv3d(conv2d.in_channels, conv2d.out_channels, \
                                kernel_size=kernel_dim, padding=padding, \
                                stride=stride, bias=conv2d.bias)

        # init the parameters of conv3d
        weight_2d = conv2d.weight.data
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2)
        weight_3d[:, :, 0, :, :] = weight_2d
        self.conv3d.weight = nn.Parameter(weight_3d)
        self.conv3d.bias = conv2d.bias

    def forward(self, x):
        out = self.conv3d(x)

        return out


##论文提到一点，说先用一个处理外观（与时间无关，卷积核为【1，3，3】）的卷积去得到外观信息，防止在时间建模中丢失外观信息，所以先捕获外观信息，后面通过残差将外观特征加上去
class APP3DC(nn.Module):
    def __init__(self, conv2d, time_dim=3, time_stride=1, temperature=4, contrastive_att=True):
        super(APP3DC, self).__init__()

        self.APM = APM(conv2d.out_channels, conv2d.out_channels//16, \
                       time_dim=time_dim, temperature=temperature, contrastive_att=contrastive_att)

        # spatial conv3d kernel
        kernel_dim = (1, conv2d.kernel_size[0], conv2d.kernel_size[1])
        stride = (1, conv2d.stride[0], conv2d.stride[0])
        padding = (0, conv2d.padding[0], conv2d.padding[1])
        self.spatial_conv3d = nn.Conv3d(conv2d.in_channels, conv2d.out_channels, \
                                        kernel_size=kernel_dim, padding=padding, \
                                        stride=stride, bias=conv2d.bias)

        # init the parameters of spatial_conv3d
        weight_2d = conv2d.weight.data
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2)
        weight_3d[:, :, 0, :, :] = weight_2d
        self.spatial_conv3d.weight = nn.Parameter(weight_3d)
        self.spatial_conv3d.bias = conv2d.bias


        # temporal conv3d kernel
        kernel_dim = (time_dim, 1, 1)
        stride = (time_stride*time_dim, 1, 1)
        padding = (0, 0, 0)
        self.temporal_conv3d = nn.Conv3d(conv2d.out_channels, conv2d.out_channels, \
                                         kernel_size=kernel_dim, padding=padding, \
                                         stride=stride, bias=False)

        # init the parameters of temporal_conv3d
        nn.init.constant_(self.temporal_conv3d.weight, 0)


    def forward(self, x):
        out = self.spatial_conv3d(x)
        residual = self.temporal_conv3d(self.APM(out))
        out = out + residual

        return out
