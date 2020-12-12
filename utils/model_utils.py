# ================================================================
#
#   Editor      : Pycharm
#   File name   : model_utils
#   Author      : HuangWei
#   Created date: 2020-12-12 10:23
#   Email       : 446296992@qq.com
#   Description : 一些涉及到模型初始化、计算损失以及AP等的函数
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
import torch

from utils.default_config import dataset_cfg


def init_weight(model: torch.nn.Module):
    """
    对于模型中的卷积层和 BN 层进行参数初始化
    如若是卷积层，则使用正态分布初始化
    如若是 BN 层，则对参数使用正态分布初始化，对偏置值使用 0 进行初始化
    该函数通过再 model.apply 中进行调用，会对模型中的每一个层进行处理

    :param model: 模型
    """
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(model.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(model.bias.data, 0.0)


def decode(conv_output: torch.Tensor, i):
    """
    将卷积最后的输出结果转化为网格
    :param conv_output: Darknet53的输出结果，三种分辨率之一
    :param i: 是哪一种分辨率
    :return: 网格
    """
    conv_shape = conv_output.shape
    batch_size = conv_shape[0]
    output_size = conv_shape[2]
    # 将网络输出的结果转换为 [batch_size, img_shape, img_shape, 3, 85]
    conv_output = torch.reshape(conv_output, [batch_size, output_size, output_size, 3, 5 + dataset_cfg.classes])

    conv_raw_dx_dy = conv_output[:, :, :, :, 0:2]  # 中心位置的偏移量
    conv_raw_dw_dh = conv_output[:, :, :, :, 2:4]  # 预测框长宽的偏移量
    conv_raw_conf = conv_output[:, :, :, :, 4:5]  # 预测框的置信度
    conv_raw_prob = conv_output[:, :, :, :, 5:]  # 预测框的类别概率

    # 好了，接下来需要画网格了。其中，output_size 等于 13、26 或者 52

    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)  # 计算网格左上角的位置

    # 根据上图公式计算预测框的中心位置
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    # 根据上图公式计算预测框的长和宽大小
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(conv_raw_conf)  # 计算预测框里object的置信度
    pred_prob = tf.sigmoid(conv_raw_prob)  # 计算预测框里object的类别概率
    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)
