# ================================================================
#
#   Editor      : Pycharm
#   File name   : load_from_txt
#   Author      : HuangWei
#   Created date: 2020-12-12 10:58
#   Email       : 446296992@qq.com
#   Description : 从 TXT 文本中读取图片的路径，然后进行加载
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
from torch.utils import data
from PIL import Image
import numpy as np
import torch

from utils.default_config import dataset_cfg
from utils.preprocess import preprocess_img, preprocess_label


class txt_loader(data.Dataset):
    """
    从文本中读取数据列表，并加载
    """

    def __init__(self, list_path, img_size=416):
        # 读取数据集中分配为训练集的txt文本，以list形式保存
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        # 读取数据集中分配为训练集的txt文本（即标签，coco数据集以txt保存 框真值），以list形式保存
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in
                            self.img_files][:12]
        self.img_files = self.img_files[:12]
        # 输入训练图像大小，即[416*416]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50  # 设定一张图像最多真实存在50个物体（封装 图像真值框时使用到）

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
        """
        需要覆写的 Dataset 函数，用于从数据集中提取单张图像以及相应的标签

        :param index: 图像的索引
        :return: 图像路径、处理后的图像tensor、坐标被归一化后的真值框
        """
        # 读取图像为tensor
        prefix = dataset_cfg.prefix
        # 拼接得到图像的绝对路径
        img_path = prefix + self.img_files[index % len(self.img_files)].rstrip()

        # 将读取的图像转化为张量
        copy_img = Image.open(img_path).copy()
        img = np.array(copy_img)

        # 处理 如若图像的通道数不为3 时(即该图像损坏)，则 读取下一张图片
        while len(img.shape) != 3:
            index += 1
            img_path = prefix + self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        # 对图像tensor进行处理（数据增强、规范化），得到：
        # 处理之后的图像， 图像的高度， 图像的宽度
        # 填充后图像的高度， 填充后图像的宽度， 填充数组
        input_img, h, w, padded_h, padded_w, pad = preprocess_img(img)

        # 对真值标签进行处理
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        label_path = prefix + label_path
        filled_labels = preprocess_label(label_path, w, h, padded_w, padded_h, pad, self.max_objects)

        # 返回 图像路径、处理后的图像tensor、坐标被归一化后的真值框filled_labels[50,5] 值在0-1之间
        return input_img, filled_labels

    def __len__(self):
        return len(self.img_files)
