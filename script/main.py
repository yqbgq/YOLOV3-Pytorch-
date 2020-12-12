# ================================================================
#
#   Editor      : Pycharm
#   File name   : main
#   Author      : HuangWei
#   Created date: 2020-12-10 14:48
#   Email       : 446296992@qq.com
#   Description : train or test model here
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
import os

import torch
from torch import optim
from torch.utils import data
from torch.backends import cudnn
from torchnet import meter

from utils.default_config import train_cfg, dataset_cfg, hyper_pars_cfg
from utils import model_utils
from model import darknet53
from dataset.load_from_txt import txt_loader


def main():
    os.makedirs('../checkpoints', exist_ok=True)  # the folders used to stored the checkpoints

    train_path_txt = dataset_cfg.train_img_list_txt  # 训练数据集存储的位置

    learning_rate = hyper_pars_cfg.learning_rate
    momentum = hyper_pars_cfg.momentum
    decay = hyper_pars_cfg.decay
    burn_in = hyper_pars_cfg.burn_in

    model = darknet53.DarkNet()
    model.apply(model_utils.init_weight)  # 对模型参数进行初始化

    if train_cfg.use_cuda:
        model = model.cuda()
        cudnn.benchmark = True
    # print(model)

    # 数据集加载器
    dataloader = torch.utils.data.DataLoader(
        txt_loader(train_path_txt),
        batch_size=train_cfg.batch_size, shuffle=False)

    # 定义了一个优化器，真正的大佬都使用 SGD，只有萌新采用 Adam，手动狗头
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=0, weight_decay=decay)

    # 计算所有损失的平均数和标准差，来统计一个epoch中损失的平均值
    loss_meter = meter.AverageValueMeter()
    previous_loss = float('inf')  # 表示正无穷
    # TODO 马上要开始写训练部分了，也就是还有训练以及损失函数的计算了


if __name__ == "__main__":
    main()
