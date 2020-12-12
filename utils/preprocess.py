# ================================================================
#
#   Editor      : Pycharm
#   File name   : preprocess
#   Author      : HuangWei
#   Created date: 2020-12-11 16:21
#   Email       : 446296992@qq.com
#   Description : 对图像进行预处理，将图像转换为 416 * 416
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
from PIL import Image
import numpy as np
from skimage.transform import resize
import torch


def preprocess_img(path):
    copy_img = Image.open(path).copy()
    img = np.array(copy_img)

    # 对图像tensor进行处理（数据增强、规范化）

    # w,h按照较大值填充成正方形
    h, w, _ = img.shape
    # np.abs 绝对值
    dim_diff = np.abs(h - w)
    # Upper (left) and lower (right) padding
    # 上（左）和下（右）填充
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding 确定填充
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    # Add padding 添加填充
    input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.

    # 填充成正方形后 resize到 指定形状（一般为416x416）
    padded_h, padded_w, _ = input_img.shape
    # Resize and normalize  resize并规范化
    input_img = resize(input_img, (416, 416, 3), mode='reflect')

    # Channels-first  转换通道
    input_img = np.transpose(input_img, (2, 0, 1))
    # As pytorch tensor  转为pytorch tensor
    input_img = torch.from_numpy(input_img).float()
    return input_img

# path = "C://Users//huangwei//Desktop//PytorchNetHub//Yolov3_pytorch//data//coco//images//train2014" \
#        "//COCO_train2014_000000000009.jpg "
# input_img = preprocess_img(path)
# print(input_img.shape)
