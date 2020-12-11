# ================================================================
#
#   Editor      : Pycharm
#   File name   : layers
#   Author      : HuangWei
#   Created date: 2020-12-11 15:07
#   Email       : 446296992@qq.com
#   Description : 
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================

from torch import nn


class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bn, activate):
        super(conv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2 if padding else 0  # padding 用 1 0 来控制是否填充
        self.bn = bn
        self.activate = activate

        self.conv_layer = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                    kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                    bias=not self.bn)

        self.bn_layer = nn.BatchNorm2d(self.out_channels)

        self.activate_layer = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.conv_layer(x)
        if self.bn:
            out = self.bn_layer(out)
        if self.activate:
            out = self.activate_layer(out)
        return out


class res_block(nn.Module):
    def __init__(self, in_channels, out_channels, bn, activate):
        super(res_block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bn = bn
        self.activate = activate

        self.conv_layer_1 = conv(in_channels=self.in_channels, out_channels=self.in_channels // 2,
                                 kernel_size=1, stride=1, padding=1,
                                 bn=self.bn, activate=self.activate)

        self.conv_layer_2 = conv(in_channels=self.in_channels // 2, out_channels=self.in_channels,
                                 kernel_size=3, stride=1, padding=1,
                                 bn=self.bn, activate=self.activate)

    def forward(self, x):
        out = self.conv_layer_1(x)
        out = self.conv_layer_2(out)

        return x + out


class up_sample(nn.Module):
    def __init__(self, in_channels, out_channels, bn):
        super(up_sample, self).__init__()

        self.conv = conv(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=1, stride=1, padding=1, bn=bn, activate=True)
        self.up_sample_layer = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        out = x
        out = self.conv(out)
        out = self.up_sample_layer(out)
        return out


class conv_sets(nn.Module):
    def __init__(self, in_channels, out_channels, bn):
        super(conv_sets, self).__init__()

        self.conv_layer_0 = conv(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=1, stride=1, padding=1, bn=bn, activate=True)

        self.conv_layer_1 = conv(in_channels=out_channels, out_channels=out_channels * 2,
                                 kernel_size=3, stride=1, padding=1, bn=bn, activate=True)
        self.conv_layer_2 = conv(in_channels=out_channels * 2, out_channels=out_channels,
                                 kernel_size=3, stride=1, padding=1, bn=bn, activate=True)

    def forward(self, x):
        out = x

        out = self.conv_layer_0(out)

        out = self.conv_layer_1(out)

        out = self.conv_layer_2(out)

        out = self.conv_layer_1(out)

        out = self.conv_layer_2(out)
        return out


class conv_yolo(nn.Module):
    def __init__(self, in_channels, class_num):
        super(conv_yolo, self).__init__()

        self.conv_layer_1 = conv(in_channels=in_channels, out_channels=in_channels * 2,
                                 kernel_size=3, stride=1, padding=1, bn=True, activate=True)
        self.conv_layer_2 = conv(in_channels=in_channels * 2, out_channels=3 * (class_num + 5),
                                 kernel_size=1, stride=1, padding=1, bn=False, activate=False)

    def forward(self, x):
        out = x

        out = self.conv_layer_1(out)

        out = self.conv_layer_2(out)

        return out
