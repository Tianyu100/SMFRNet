from typing import Union, List
import torch.nn.functional as F
import torch
import torch.nn as nn
# from Deform import *
# from GatedSpatialConv import MyGatedSpatialConv2d
import torch.utils.data as Data



class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DownConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)  #dilation = 1 就是普通卷积
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        return self.relu(self.bn(self.conv(x)))


class UpConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        return self.relu(self.bn(self.conv(torch.cat([x1, x2], dim=1))))


class RSU(nn.Module):
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()   #height代表卷积块（conv+bn+relu）的(深度)个数,

        assert height >= 2
        self.conv_in = ConvBNReLU(in_ch, out_ch)

        encode_list = [DownConvBNReLU(out_ch, mid_ch, flag=False)]
        decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]
        for i in range(height - 2):
            encode_list.append(DownConvBNReLU(mid_ch, mid_ch))
            decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))

        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))
        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()   #pop()默认删除最后一个列表值
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(x, x2)

        return x + x_in


class RSU4F(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.encode_modules = nn.ModuleList([ConvBNReLU(out_ch, mid_ch),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=1),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=5)])

        self.decode_modules = nn.ModuleList([ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch * 2, mid_ch, dilation=5),
                                             ConvBNReLU(mid_ch * 2, out_ch)])
#         self.encode_modules = nn.ModuleList([ConvBNReLU(out_ch, mid_ch),
#                                              ConvBNReLU(mid_ch, mid_ch, dilation=2),
#                                              ConvBNReLU(mid_ch, mid_ch, dilation=4),
#                                              ConvBNReLU(mid_ch, mid_ch, dilation=8)])

#         self.decode_modules = nn.ModuleList([ConvBNReLU(mid_ch * 2, mid_ch, dilation=4),
#                                              ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
#                                              ConvBNReLU(mid_ch * 2, out_ch)])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))

        return x + x_in




# 搭建dla的基础模块之一，可选项   可使用2次膨胀卷积
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False,
                               dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation)
        # self.conv2 = DeformConv2d(planes, planes, kernel_size=3, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out
# dla的基础模块，作用就是聚合多个输入张量，先通道维度拼接，然后加入卷积+BN，可选短链接
class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x
# 整个dla最难看懂的部分，实现在level或stage内部的各种特征融合，levels=1,2,3(非level)分别对应论文中示意图的三个大小不同的红色方框，分别是1,2,3级结构
# levels=1 可以看到是两个卷积模块的串接结果，一定注意是串接不是上下层关系，串接输出就是最终输出，构成1级结构
# levels=2 可以看到是两个1级结构的串接结果，一定注意是串接不是上下层关系，串接输出就是最终输出，构成2级结构
# levels=3 可以看到是两个2级结构的串接结果，一定注意是串接不是上下层关系，串接输出就是最终输出，构成3级结构
# 从2级3级结构上来看，当各子级结构串接时，前面的子级结构相比接在后面的子级结构少了两个输入，或者说前面的子级结构的有两个输入是空，这样前后子级结构保持一致
class Tree(nn.Module):  # levels是几级子结构，block是基础模块，level_root判断有没有IDA结构，即示意图中的黄色连接线
    """ self.level2 = Tree(levels[2]=1, block, channels[1], channels[2], 2, level_root=False, root_residual=residual_root)
        self.level3 = Tree(levels[3]=2, block, channels[2], channels[3], 2, level_root=True, root_residual=residual_root)
"""
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1, dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels, stride, root_dim=0,
                              root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels, root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
            # self.downsample = GELayers2(in_channels,in_channels)  #替换为卷积残差块，进行下采样
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x  #[_,32,128,_]
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)  # 示意图中的绿色聚合点都是root模块，这个不是太好想明白
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x
class DLAUnet(nn.Module):
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int, levels=[1, 1, 1, 2, 1],
                 block=BasicBlock, residual_root=False,linear_root=False):
        super().__init__()  # height代表卷积块（conv+bn+relu）的(深度)个数,

        #channels= [16, 32, 64, 128, 256, 512]     64, 16, 128

        self.level0 = self._make_conv_level(in_ch, out_ch, levels[0])
        self.level1 = self._make_conv_level(out_ch,mid_ch, levels[1])
        self.level2 = Tree(levels[2], block, mid_ch, 2*mid_ch, 2, level_root=False,#下采样一次 pool(k=2,s=2)
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, 2*mid_ch,4*mid_ch,2, level_root=True,   # 4*mid_ch=out_ch=64
                           root_residual=residual_root)  #下采样一次 pool(k=2,s=2)
        self.level4 = Tree(levels[4], block,4*mid_ch,8*mid_ch,2, level_root=False, root_residual=residual_root)
        # 下采样一次 pool(k=2,s=2)
        self.level5=self._make_conv_level(8*mid_ch, 8*mid_ch,levels[0],dilation=2)

        # encode
        encode_list = []
        for i in range(6):
            encode_list.append(getattr(self, 'level{}'.format(i)))

        # decode
        decode_list = [UpConvBNReLU(2*8*mid_ch, 4*mid_ch,flag=False)]
        decode_list.append(UpConvBNReLU(2*4*mid_ch, 2*mid_ch))
        decode_list.append(UpConvBNReLU(2*2*mid_ch,mid_ch))
        decode_list.append(UpConvBNReLU(2*mid_ch,out_ch))

        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.encode_modules[0](x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules[1:]:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()  # pop()默认删除最后一个列表值
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(x, x2)
        return x + x_in

        # 常规卷积模块，用于构建level0和level1
    def _make_conv_level(self, inplanes, planes, convs,kernel_size=3, stride=1, dilation=1):
        """ self.level0=self._make_conv_level(channels[0], channels[0], levels[0])
            self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)"""
        modules = []

        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride if i == 0 else 1, padding=dilation, bias=False,
                          dilation=dilation),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)


class IncepGCN(nn.Module):
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=1/4,dilation=2):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel number of a convolution branch

        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)

        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)

        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.split_indexes = (gc, gc, gc, in_channels - 3 * gc)

        self.conv1x1 = nn.Conv2d(in_channels, in_channels, 1, bias=True)


        padding = 1 if dilation == 1 else dilation
        self.convd = nn.Conv2d(in_channels, in_channels, 3, padding=padding, dilation=dilation, bias=True)


        self.bn = nn.BatchNorm2d(in_channels)
        self.relu=nn.ReLU(inplace=True)

    def forward(self, x):
        # B, C, H, W = x.shape
        x_hw, x_w, x_h, x_id = torch.split(x, self.split_indexes, dim=1)
        x=torch.cat(
            (self.dwconv_hw(x_hw),
             self.dwconv_w(x_w),
             self.dwconv_h(x_h),
             x_id),
            dim=1)
        x=self.relu(self.bn(x))
        x=self.conv1x1(x)
        x=self.convd(x)
        x=self.relu(x)
        return x
class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = self.dilate1(x)
        dilate2_out = self.conv1x1(self.dilate2(x))
        dilate3_out = self.conv1x1(self.dilate2(self.dilate1(x)))
        dilate4_out = self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out
class MyDACblock(nn.Module):
    def __init__(self, channel):
        super(MyDACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        # self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = self.dilate1(x)              #x
        dilate2_out = self.dilate2(dilate1_out)    #x->d=2
        dilate3_out = self.dilate3(dilate1_out)    #x->d=5
        dilate4_out = self.dilate3(dilate2_out)    #x->d=2->d=5
        out = dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class RRB_Block(nn.Module):
    def __init__(self, in_channels, out_channels,stride=1,dilation=1,ration=1):
        super(RRB_Block, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=stride,
                              bias=True,dilation=dilation)

        self.conv1 = nn.Conv2d(out_channels,ration*out_channels, kernel_size=3, stride=stride,
                              padding=1,bias=False,dilation=dilation)

        self.conv2 = nn.Conv2d(ration*out_channels,out_channels, kernel_size=3, stride=stride,
                             padding=1,bias=False, dilation=dilation)

        # self.convDeform=DeformConv2d(ratio*out_channels, out_channels, kernel_size=3, stride=1, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(ration*out_channels)


        self.relu = nn.ReLU(inplace=True)

        self._initialize_weights(self)

    def _initialize_weights(*models):
        for model in models:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1.)
                    m.bias.data.fill_(1e-4)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, 0.0001)
                    m.bias.data.zero_()

    def forward(self, x):
        x=self.bn(self.conv1x1(x))
        residual = x

        x=self.conv1(x)
        x=self.relu(self.bn1(x))
        x=self.conv2(x)
        # x=self.convDeform(x)

        x=self.bn(x)+residual
        x=self.relu(x)
        return x


def GE_ConvBNReLU(in_chann, out_chann, ks=3, st=1, p=1, with_act=True):
    return nn.Sequential(
        nn.Conv2d(in_chann, out_chann, ks, st, p, bias=False),
        nn.BatchNorm2d(out_chann),
        nn.ReLU(inplace=True)
        )
class GELayers2(nn.Module):
    def __init__(self, in_chann, out_chann, padding=1, dilation=1):
        super(GELayers2, self).__init__()
        mid_chann = in_chann * 2
        self.conv1 = GE_ConvBNReLU(in_chann, in_chann, 3, 1, 1)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(in_chann, mid_chann, kernel_size=3, stride=2,
                      padding=1, groups=in_chann, bias=False),
            nn.BatchNorm2d(mid_chann),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(mid_chann, mid_chann, kernel_size=3, stride=1,
                      padding=padding, groups=mid_chann, dilation=dilation, bias=False),
            nn.BatchNorm2d(mid_chann),
            # SEModule(mid_chann),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_chann, out_chann, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chann)
        )

        self.short = nn.Sequential(
            nn.Conv2d(in_chann, in_chann, 3, 2, 1, groups=in_chann,
                      bias=False),
            nn.BatchNorm2d(in_chann),
            nn.Conv2d(in_chann, out_chann, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chann)
        )


        self.relu = nn.ReLU(inplace=True)
        self._init_weights()


    def _init_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)

        short=self.short(x)
        out = feat + short
        return self.relu(out)



class GCN_Block(nn.Module):
    def __init__(self,k_size,in_channels,out_channels,stride=1,dilation=1):
        super(GCN_Block, self).__init__()
        self.kernel_size=k_size

        self.conv0 = nn.Conv2d(in_channels,in_channels, kernel_size=3, stride=stride,
                               padding=dilation,bias=False,dilation=dilation)
        self.bn1 = nn.BatchNorm2d(in_channels)

        assert self.kernel_size % 2 == 1, 'Kernel size must be odd'
        self.conv11 = nn.Conv2d(in_channels, out_channels,
                                kernel_size=(self.kernel_size, 1), padding=(self.kernel_size//2, 0))
        self.conv12 = nn.Conv2d(out_channels, out_channels,
                                kernel_size=(1, self.kernel_size), padding=(0, self.kernel_size//2),bias=False)

        self.conv21 = nn.Conv2d(in_channels, out_channels,
                                kernel_size=(1, self.kernel_size), padding=(0, self.kernel_size//2))
        self.conv22 = nn.Conv2d(out_channels, out_channels,
                                kernel_size=(self.kernel_size, 1), padding=(self.kernel_size//2, 0),bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # self.convDeform=DeformConv2d(out_channels, out_channels, kernel_size=3, stride=1, bias=False)

        self.convdown=None
        if in_channels!=out_channels:
            self.convdown = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                                   padding=dilation, bias=False, dilation=dilation)
        self._initialize_weights(self)

    def _initialize_weights(*models):
        for model in models:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1.)
                    m.bias.data.fill_(1e-4)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, 0.0001)
                    m.bias.data.zero_()

    def forward(self, x):

        residual = self.convdown(x) if self.convdown else x

        x=self.conv0(x)
        x=self.relu(self.bn1(x))

        x1 = self.conv11(x)
        x1 = self.conv12(x1)
        x1=self.relu(self.bn2(x1))

        x2 = self.conv21(x)
        x2 = self.conv22(x2)
        x2 = self.bn2(x2)

        x12 = x1 + x2

        # x=self.convDeform(x12)
        # x=x+x12+residual
        x=x12+residual
        x=self.relu(x)
        return x



class DeepPoolLayer(nn.Module):  #FAM
    """FAM & FPN FUSE操作"""

    def __init__(self, k, k_out, need_x2=False, need_fuse=False):
        """
        k: FAM输入的通道数
        k_out: FAM输出的通道数
        need_x2: 是否有backbone第i-1个金字塔层(从左到右编号)
        need_fuse: 是否有GGF的输出
        """
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [2, 4, 8]  # 下采样比例
        self.need_x2 = need_x2
        self.need_fuse = need_fuse
        pools, convs = [], []
        for i in self.pools_sizes:
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
        self.pools = nn.ModuleList(pools)  # 平均下采样/平均池化
        self.convs = nn.ModuleList(convs)  # 3×3卷积层，有padding所以特征尺寸不会改变
        self.relu = nn.ReLU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)  # FAM中sum后的3×3卷积，有padding
        if self.need_fuse:
            self.conv_sum_c = nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False)  # FPN操作中的3×3卷积，有padding

    def forward(self, x, x2=None, x3=None):
        """
        merge = self.deep_pool[0](conv2merge[0], conv2merge[1], infos[0])
        x: backbone第i个金字塔层的输出（假设从左到右/从浅到深编号）
        x2: backbone第i-1个金字塔层的输出
        x3 : 对应GGF的输出
        """
        x_size = x.size()
        # FAM
        resl = x
        for i in range(len(self.pools_sizes)):
            y = self.convs[i](self.pools[i](x))  # 将输入先后进行平均池化、3×3卷积
        #     resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))  # 上采样后4个分支相加
        resl = self.relu(resl)  # ReLU

        resl = self.conv_sum(resl)  # FAM中sum操作后的3×3卷积
        return resl

class PyramidPool(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size):
        super(PyramidPool, self).__init__()
        modules=[]
        self.pool_moudles=[]
        for i in pool_size:
            layer=nn.Sequential(
                nn.AdaptiveAvgPool2d(i),
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels, momentum=0.95),
                nn.ReLU(inplace=True)
            )
            modules.append(layer)
        self.pool_moudles=nn.ModuleList(modules)

        self.final = nn.Sequential(
            nn.Conv2d(3072, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
        )
    def forward(self, x):
        size = x.shape
        x1=self.pool_moudles[0](x)
        x2=self.pool_moudles[1](x)
        x3=self.pool_moudles[2](x)
        x4=self.pool_moudles[3](x)

        out=torch.cat([
            x,
            F.upsample_bilinear(x1,size[2:]),
            F.upsample_bilinear(x2,size[2:]),
            F.upsample_bilinear(x3,size[2:]),
            F.upsample_bilinear(x4,size[2:]),
        ], dim=1)
        out=self.final(out)
        return out

class PyramidConv(nn.Module):
    def __init__(self, inplans, planes, pyconv_kernels=[5, 7], stride=1, pyconv_groups=[8, 16]):
        super(PyramidConv, self).__init__()
        self.conv0=nn.Conv2d(inplans,planes,3,stride=1,padding=1,dilation=1)
        self.bn0=nn.BatchNorm2d(planes)

        self.conv2_1 = nn.Conv2d(planes, planes, kernel_size=pyconv_kernels[0], stride=stride,
                                 padding=pyconv_kernels[0] // 2, dilation=1, groups=pyconv_groups[0], bias=False)
        self.bn1=nn.BatchNorm2d(planes)

        self.conv2_2 = nn.Conv2d(planes, planes, kernel_size=pyconv_kernels[1], stride=stride,
                                 padding=pyconv_kernels[1] // 2, dilation=1, groups=pyconv_groups[1], bias=False)
        self.bn2=nn.BatchNorm2d(planes)

        self.Gelu=nn.GELU()


    def forward(self,x):
        x0=self.bn0(self.conv0(x))
        x0_out = self.Gelu(x0)

        x1=self.conv2_1(x0) #conv3x3->5x5
        x1_out=self.Gelu(self.bn1(x1))

        x2=self.conv2_2(x1)  #conv3x3->5x5->7x7
        x2_out =self.Gelu(self.bn2(x2))
        x_out=x0_out +x1_out+x2_out

        # return torch.cat((x0_out,  x1_out, x2_out), dim=1)
        return x_out

class RRB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RRB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.relu(self.bn(res))
        res = self.conv3(res)
        return self.relu(x + res)

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # x 的输入格式是：[batch_size, C, H, W]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class U2Net(nn.Module):
    def __init__(self, cfg: dict, out_ch: int = 1):
        super().__init__()
        assert "encode" in cfg
        self.encode_num = len(cfg["encode"])


        #-----------------encode+side -----------------
        encode_list = []

        for num,c in enumerate(cfg["encode"]):
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) == 6
            encode_list.append(DLAUnet(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))
        #执行init()时，在init里构建好模型，比如执行U2Net的init，第一次循环，先跳入RSU,在RSU类继续执行他的init，然后返回到这里
        #接着执行下一次循环，创建一个RSU块
        #只有执行到U2Net的forward时，将输入传给指定RSU模块，再去执行RSU的forward函数,实际就是给RSU实例化
        self.encode_modules = nn.ModuleList(encode_list)

        # self.FAM5=DeepPoolLayer(512,512)
        # self.FAM4=DeepPoolLayer(256,256)
        # self.FAM3=DeepPoolLayer(128,128)
        # self.FAM2=DeepPoolLayer(64,64)
        # self.FAM1=DeepPoolLayer(64,64)


        # self.atten_edge5=MyGatedSpatialConv2d(512,512,ratio=1)
        # self.atten_edge4=MyGatedSpatialConv2d(512,512,ratio=1)
        # self.atten_edge3=MyGatedSpatialConv2d(256,256,ratio=2)
        # self.atten_edge2=MyGatedSpatialConv2d(128,128,ratio=2)
        # self.atten_edge1=MyGatedSpatialConv2d(64,64,ratio=4)

        # ## -------------Decoder--------------
        # stage 5d
        # self.conv6d_1 = nn.Conv2d(1024, 512, 3, padding=1)  # 16
        # self.bn6d_1 = nn.BatchNorm2d(512)
        # self.relu6d_1 = nn.ReLU(inplace=True)

        self.conv6d_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bn6d_2 = nn.BatchNorm2d(512)
        self.relu6d_2 = nn.ReLU(inplace=True)

        # #将两个d=2卷积块转成d=1,2,5金字塔块
        # self.DAC=MyDACblock(512)
        # self.bn6d = nn.BatchNorm2d(512)
        # self.relu6d = nn.ReLU(inplace=True)

        # 改成金字塔池化
        # self.conv6d_1 = nn.Conv2d(1024, 512, 3, padding=1)  # 16
        # self.bn6d_1 = nn.BatchNorm2d(512)
        # self.relu6d_1 = nn.ReLU(inplace=True)
        # self.PPool=PyramidPool(1024,512, pool_size=[1,2,3,6])

        # stage 4d
        self.conv5d_1 = nn.Conv2d(512 + 512,256, 3, padding=1)  # 16
        self.bn5d_1 = nn.BatchNorm2d(256)
        self.relu5d_1 = nn.ReLU(inplace=True)

        self.conv5d_m = nn.Conv2d(256, 512, 3, padding=1)  ###
        self.bn5d_m = nn.BatchNorm2d(512)
        self.relu5d_m = nn.ReLU(inplace=True)

        # stage 3d
        self.conv4d_1 = nn.Conv2d(512+512,256, 3, padding=1)  # 32
        self.bn4d_1 = nn.BatchNorm2d(256)
        self.relu4d_1 = nn.ReLU(inplace=True)

        self.conv4d_m = nn.Conv2d(256, 256, 3, padding=1)  ###
        self.bn4d_m = nn.BatchNorm2d(256)
        self.relu4d_m = nn.ReLU(inplace=True)

        self.conv4d_2 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4d_2 = nn.BatchNorm2d(512)
        self.relu4d_2 = nn.ReLU(inplace=True)

        # stage 2d
        self.conv3d_1 = nn.Conv2d(256+256, 128, 3, padding=1)  # 64
        self.bn3d_1 = nn.BatchNorm2d(128)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.conv3d_m = nn.Conv2d(128, 128, 3, padding=1)  ###
        self.bn3d_m = nn.BatchNorm2d(128)
        self.relu3d_m = nn.ReLU(inplace=True)

        self.conv3d_2 = nn.Conv2d(128,256,3, padding=1)
        self.bn3d_2 = nn.BatchNorm2d(256)
        self.relu3d_2 = nn.ReLU(inplace=True)

        # stage 1d
        self.conv2d_1 = nn.Conv2d(128 + 128, 64, 3, padding=1)  # 128
        self.bn2d_1 = nn.BatchNorm2d(64)
        self.relu2d_1 = nn.ReLU(inplace=True)

        self.conv2d_m = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2d_m = nn.BatchNorm2d(64)
        self.relu2d_m = nn.ReLU(inplace=True)

        self.conv2d_2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2d_2 = nn.BatchNorm2d(128)
        self.relu2d_2 = nn.ReLU(inplace=True)

        # stage 1d
        self.conv1d_1 = nn.Conv2d(64 + 64, 64, 3, padding=1)  # 128
        self.bn1d_1 = nn.BatchNorm2d(64)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # self.conv1d_m = nn.Conv2d(64, 64, 3, padding=1)
        # self.bn1d_m = nn.BatchNorm2d(64)
        # self.relu1d_m = nn.ReLU(inplace=True)

        self.conv1d_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1d_2 = nn.BatchNorm2d(64)
        self.relu1d_2 = nn.ReLU(inplace=True)


        #PAC金字塔卷积
        self.PAC6=PyramidConv(512,512)
        self.PAC5=PyramidConv(512,512)
        self.PAC4=PyramidConv(512,512)
        self.PAC4_1=PyramidConv(512,256)
        self.PAC3=PyramidConv(256,256)
        self.PAC3_1=PyramidConv(256,128)
        self.PAC2=PyramidConv(128,128)
        self.PAC2_1=PyramidConv(128,64)
        self.PAC1=PyramidConv(64,64)

        #CBAM 
        self.SE56 = ChannelAttention(512)
        self.SE45 = ChannelAttention(512)
        self.CBAM34=CBAMLayer(256)
        self.CBAM23=CBAMLayer(128)
        self.CBAM12=CBAMLayer(64)


        #RRB
        self.RRB56=RRB(512,512)
        self.RRB45=RRB(512,512)
        self.RRB34=RRB(256,256)
        self.RRB23=RRB(128,128)
        self.RRB12=RRB(64,64)


        ## -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32, mode='bilinear')  ###
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        ## -------------Side Output--------------
        self.outconv6 = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv5 = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv4 = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv3 = nn.Conv2d(256, 1, 3, padding=1)
        self.outconv2 = nn.Conv2d(128, 1, 3, padding=1)
        self.outconv1 = nn.Conv2d(64, 1, 3, padding=1)

        self.out_conv = nn.Conv2d(6, 1, kernel_size=1)


    def make_layer(self, inplanes, planes,dilation):
        """ self.level0=self._make_conv_level(channels[0], channels[0], levels[0])
            self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)"""
        modules = []

        for i in dilation:
            modules.extend([
                RRB_Block(inplanes, planes),
                IncepGCN(planes,dilation=i)],

            )
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        _, _, h,w = x.shape

        # ------encode--------
        encode_outputs = []
        encode_outputs1 = []
        for i, m in enumerate(self.encode_modules):
            x = m(x)
            encode_outputs.append(x)
            if i < 5:
                x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)  #m1,2,3都跟pooling

        m6 = encode_outputs[5]
        m5 = encode_outputs[4]
        m4 = encode_outputs[3]
        m3 = encode_outputs[2]
        m2 = encode_outputs[1]
        m1 = encode_outputs[0]

        ## -------------Boundary refine-------------
        m6=self.PAC6(m6)
        upm6 = self.upscore2(m6)  # 8 -> 16
        m5=self.PAC5(m5)
        fuse56=upm6*m5
        skip56=self.SE56(upm6+fuse56)


        m5=m5+skip56+fuse56
        m5 = self.PAC5(m5)
        upm5 = self.upscore2(m5)  # 16 -> 32
        m4=self.PAC4(m4)
        fuse45=upm5*m4
        skip45=self.SE45(upm5+fuse45)


        m4=m4+skip45+fuse45
        h4 = self.PAC4_1(m4)
        upm4 = self.upscore2(h4)  # 32 -> 64
        m3 = self.PAC3(m3)
        fuse34=upm4*m3
        skip34 =self.CBAM34(upm4 + fuse34)

        m3=m3+skip34+fuse34
        h3 = self.PAC3_1(m3)
        upm3 = self.upscore2(h3)  # 64 -> 128
        m2 = self.PAC2(m2)
        fuse23=upm3*m2
        skip23 =self.CBAM23(upm3 + fuse23)

        m2 = m2 + skip23+fuse23
        h2= self.PAC2_1(m2)
        upm2 = self.upscore2(h2)  # 128 -> 256
        m1 = self.PAC1(m1)
        fuse12=upm2 *m1
        skip12 =self.CBAM12(upm2 + fuse12)

        m1=m1+skip12+fuse12
        m1=self.PAC1(m1)

        #---------Decoder-----------
        rrbm6=self.RRB56(upm6)
        rrbm6 = self.relu6d_2(self.bn6d_2(self.conv6d_2(rrbm6)))

        m5 = self.relu5d_1(self.bn5d_1(self.conv5d_1(torch.cat((rrbm6, m5), 1))))
        m5 = self.relu5d_m(self.bn5d_m(self.conv5d_m(m5)))  # 2个常规卷积

        rrbm5=self.RRB45(upm5)
        m4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((rrbm5, m4), 1))))
        m4 = self.relu4d_m(self.bn4d_m(self.conv4d_m(m4)))  # 2个常规卷积，不改变特征图大小
        m4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(m4)))  # 2个常规卷积，不改变特征图大小

        rrbm4 = self.RRB34(upm4)
        m3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((rrbm4, m3), 1))))
        m3 = self.relu3d_m(self.bn3d_m(self.conv3d_m(m3)))  # 2个常规卷积，不改变特征图大小
        m3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(m3)))  # 2个常规卷积，不改变特征图大小

        rrbm3 =self.RRB23(upm3)
        m2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((rrbm3,m2), 1))))
        m2 = self.relu2d_m(self.bn2d_m(self.conv2d_m(m2)))
        m2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(m2)))
        # hd1 = self.FAM1(hx)

        rrbm2 = self.RRB12(upm2)
        m1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((rrbm2, m1), 1))))
        # m1 = self.relu2d_m(self.bn2d_m(self.conv2d_m(m2)))
        m1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(m1)))
        # hd1 = self.FAM1(hx)




        ## -------------Side Output-------------

        d6 = self.outconv6(m6)
        d6 = self.upscore6(d6)  # 8->256

        d5 = self.outconv5(m5)
        d5 = self.upscore5(d5)  # 16->256

        d4 = self.outconv4(m4)
        d4 = self.upscore4(d4)  # 32->256

        d3 = self.outconv3(m3)
        d3 = self.upscore3(d3)  # 64->256

        d2 = self.outconv2(m2)
        d2 = self.upscore2(d2)  # 128->256

        d1 = self.outconv1(m1)  # 256

        side_outputs = [d1,d2,d3,d4,d5,d6]
        out=self.out_conv(torch.concat(side_outputs, dim=1))


        if self.training:
            # do not use torch.sigmoid(0~1) for amp safe  防止计算loss时，出现了除以0的情况
            return [torch.sigmoid(out)]+[torch.sigmoid(i) for i in side_outputs]
#             return [out]+[i for i in side_outputs]

        else:
            return torch.sigmoid(out)



def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()



def u2net_full(out_ch: int = 1):
    cfg = {             #cfg是个字典
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 32, 64, False, False],      # En1
                   [6, 64, 32, 128, False, False],    # En2
                   [5, 128, 64, 256, False, False],   # En3
                   [4, 256, 128, 512, False, False],  # En4
                   [4, 512, 256, 512, True, False],   # En5
                   [4, 512, 256, 512, True, True]],   # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 1024, 256, 512, True, True],   # De5
                   [4, 1024, 128, 256, False, True],  # De4
                   [5, 512, 64, 128, False, True],    # De3
                   [6, 256, 32, 64, False, True],     # De2
                   [7, 128, 16, 64, False, True]]     # De1
    }

    return U2Net(cfg, out_ch)


def u2net_lite(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[0, 3, 32, 64, False, False],  # En1
                   [0, 64, 32,128, False,False],  # En2
                   [0, 128,64,256, False, True],  # En3
                   [0, 256,64,512, False, True],  # En4
                   [4, 512, 256, 512, True,True],  # En5
                   [4, 512, 256, 512, True, True],] # En6

         }

    return U2Net(cfg, out_ch)

def convert_onnx(m, save_path):
    m.eval()
    x = torch.rand(1, 3, 288, 288, requires_grad=True)

    # export the model
    torch.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      save_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,
                      opset_version=11)


if __name__ == '__main__':
    # n_m = RSU(height=7, in_ch=3, mid_ch=12, out_ch=3)
    # convert_onnx(n_m, "RSU7.onnx")
    #
    # n_m = RSU4F(in_ch=3, mid_ch=12, out_ch=3)
    # convert_onnx(n_m, "RSU4F.onnx")

    u2net = u2net_lite()
    # print(u2net)
    # convert_onnx(u2net, "u2net_full.onnx")

    x=torch.randn([2,3,256,256])
    out=u2net(x)
    print(out)


