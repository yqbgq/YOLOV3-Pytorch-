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
