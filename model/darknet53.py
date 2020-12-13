# ================================================================
#
#   Editor      : Pycharm
#   File name   : darknet53
#   Author      : HuangWei
#   Created date: 2020-12-11 14:27
#   Email       : 446296992@qq.com
#   Description : 这里实现了 DarkNet 53
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================

import torch

from .layers import *


class DarkNet(nn.Module):
    def __init__(self, input_size=416):
        super(DarkNet, self).__init__()
        self.input_size = input_size

        self.__build_module()

    def __build_module(self):
        self.module_1 = nn.ModuleList()
        self.module_2 = nn.ModuleList()
        self.module_3 = nn.ModuleList()

        self.module_1.append(conv(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1,
                                  bn=True, activate=True))

        self.module_1.append(conv(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1,
                                  bn=True, activate=True))

        self.module_1.append(res_block(in_channels=64, out_channels=64, bn=True, activate=True))

        self.module_1.append(conv(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1,
                                  bn=True, activate=True))

        for i in range(2):
            self.module_1.append(res_block(in_channels=128, out_channels=128, bn=True, activate=True))

        self.module_1.append(conv(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1,
                                  bn=True, activate=True))

        for i in range(8):
            self.module_1.append(res_block(in_channels=256, out_channels=256, bn=True, activate=True))

        self.module_2.append(conv(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1,
                                  bn=True, activate=True))

        for i in range(8):
            self.module_2.append(res_block(in_channels=512, out_channels=512, bn=True, activate=True))

        self.module_3.append(conv(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1,
                                  bn=True, activate=True))

        for i in range(4):
            self.module_3.append(res_block(in_channels=1024, out_channels=1024, bn=True, activate=True))

        self.conv_set_1 = conv_sets(in_channels=1024, out_channels=512, bn=True)

        self.conv_yolo_1 = conv_yolo(in_channels=512, class_num=80)

        self.up_sample_1 = up_sample(in_channels=512, out_channels=256, bn=True)

        self.conv_set_2 = conv_sets(in_channels=768, out_channels=256, bn=True)

        self.conv_yolo_2 = conv_yolo(in_channels=256, class_num=80)

        self.up_sample_2 = up_sample(in_channels=256, out_channels=128, bn=True)

        self.conv_set_3 = conv_sets(in_channels=384, out_channels=128, bn=True)

        self.conv_yolo_3 = conv_yolo(in_channels=128, class_num=80)

    def forward(self, x):
        out_data = x

        for layers in self.module_1:
            out_data = layers(out_data)
            # print(out_data.shape)

        route_1 = out_data

        for layers in self.module_2:
            out_data = layers(out_data)
            # print(out_data.shape)

        route_2 = out_data

        for layers in self.module_3:
            out_data = layers(out_data)
            # print(out_data.shape)

        route_3 = out_data

        return self.get_yolo_data(route_1, route_2, route_3)

    def get_yolo_data(self, r1, r2, r3):
        r3 = self.conv_set_1(r3)
        little_yolo_data = self.conv_yolo_1(r3)
        r3 = self.up_sample_1(r3)

        r2 = torch.cat([r3, r2], dim=1)
        r2 = self.conv_set_2(r2)
        middle_yolo_data = self.conv_yolo_2(r2)
        r2 = self.up_sample_2(r2)

        r1 = torch.cat([r2, r1], dim=1)
        r1 = self.conv_set_3(r1)
        large_yolo_data = self.conv_yolo_3(r1)

        return large_yolo_data, middle_yolo_data, little_yolo_data



