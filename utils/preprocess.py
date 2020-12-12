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
import os

from PIL import Image
import numpy as np
from skimage.transform import resize
import torch


def preprocess_img(img: np.ndarray):
    # TODO 这里是直接对大佬的代码进行复制处理，还没有怎么仔细理解
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

    return input_img, h, w, padded_h, padded_w, pad


def preprocess_label(label_path, w, h, padded_w, padded_h, pad, max_objects):
    # TODO 这里是直接对大佬的代码进行复制处理，还没有怎么仔细理解
    labels = None
    if os.path.exists(label_path):
        # eg：[8,5]   8：该图像有8个bbox   5: 0代表类别对应序号 1~4代表坐标（值在0~1之间）
        labels = np.loadtxt(os.path.abspath(label_path)).reshape(-1, 5)
        # Extract coordinates for unpadded + unscaled image
        # 提取未填充+未缩放图像的坐标
        x1 = w * (labels[:, 1] - labels[:, 3] / 2)
        y1 = h * (labels[:, 2] - labels[:, 4] / 2)
        x2 = w * (labels[:, 1] + labels[:, 3] / 2)
        y2 = h * (labels[:, 2] + labels[:, 4] / 2)
        # Adjust for added padding
        # 添加填充，以便于 图像调整一致
        x1 += pad[1][0]
        y1 += pad[0][0]
        x2 += pad[1][0]
        y2 += pad[0][0]
        # Calculate ratios from coordinates
        # 从坐标计算比率
        labels[:, 1] = ((x1 + x2) / 2) / padded_w
        labels[:, 2] = ((y1 + y2) / 2) / padded_h
        labels[:, 3] *= w / padded_w
        labels[:, 4] *= h / padded_h
    # Fill matrix
    # 填充矩阵（将 txt里的内容，即每张图像的所有物体填入，最多添加50个物体）
    filled_labels = np.zeros((max_objects, 5))
    if labels is not None:
        filled_labels[range(len(labels))[:max_objects]] = labels[:max_objects]
    filled_labels = torch.from_numpy(filled_labels)


def preprocess_img_from_path(path):
    copy_img = Image.open(path).copy()
    img = np.array(copy_img)

    input_img = preprocess_img(img)

    return input_img

# path = "C://Users//huangwei//Desktop//PytorchNetHub//Yolov3_pytorch//data//coco//images//train2014" \
#        "//COCO_train2014_000000000009.jpg "
# input_img = preprocess_img(path)
# print(input_img.shape)
