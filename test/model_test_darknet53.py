# ================================================================
#
#   Editor      : Pycharm
#   File name   : test_darknet53
#   Author      : HuangWei
#   Created date: 2020-12-11 18:22
#   Email       : 446296992@qq.com
#   Description : 测试一下darknet53网络的输出
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
import torch

from model import darknet53
from utils import preprocess

path = "C://Users//huangwei//Desktop//PytorchNetHub//Yolov3_pytorch//data//coco//images//train2014" \
       "//COCO_train2014_000000000009.jpg "

dn = darknet53.DarkNet()

input_img = preprocess.preprocess_img(path)  # type: torch.Tensor
input_img = torch.reshape(input_img, [1, 3, 416, 416])
print("input data shape", input_img.shape)  # input data shape torch.Size([1, 3, 416, 416])

r_1, r_2, r_3 = dn(input_img)

"""
r_1: torch.Size([1, 256, 52, 52])
r_2: torch.Size([1, 512, 26, 26])
r_3: torch.Size([1, 1024, 13, 13])
"""
print("r_1:", r_1.shape)
print("r_2:", r_2.shape)
print("r_3:", r_3.shape)

little, middle, large = dn.get_yolo_data(r_1, r_2, r_3)

"""
分别对应三种不同分辨率的特征图
little: torch.Size([1, 255, 13, 13])
middle: torch.Size([1, 255, 26, 26])
large: torch.Size([1, 255, 52, 52])
"""
print("little:", little.shape)
print("middle:", middle.shape)
print("large:", large.shape)

print("OK!")
