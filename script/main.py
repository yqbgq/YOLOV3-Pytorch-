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
import numpy as np

from utils.default_config import train_cfg, dataset_cfg, hyper_pars_cfg
from utils import model_utils
from model import darknet53
from dataset.load_from_txt import txt_loader

anchors = torch.Tensor(
    [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
)
feature_shape = [52, 26, 13]


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
    model.train()

    # 数据集加载器
    dataloader = torch.utils.data.DataLoader(
        txt_loader(train_path_txt),
        batch_size=train_cfg.batch_size, shuffle=False)

    # 定义了一个优化器，真正的大佬都使用 SGD，只有萌新采用 Adam，手动狗头
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=0, weight_decay=decay)

    # 计算所有损失的平均数和标准差，来统计一个epoch中损失的平均值
    loss_meter = meter.AverageValueMeter()
    previous_loss = float('inf')  # 表示正无穷

    for step in range(train_cfg.epochs):
        # 清空仪表信息和混淆矩阵信息
        loss_meter.reset()
        # 每轮epoch
        for batch_i, (images, targets) in enumerate(dataloader):
            # images :处理后的图像tensor[16,3,416,416]        targets:坐标被归一化后的真值框filled_labels[16,50,5] 值在0-1之间
            if train_cfg.use_cuda:
                images = images.cuda()
                targets = targets.cuda()

            # 优化器梯度清零
            optimizer.zero_grad()
            # 得到网络输出值，作为损失 (loss :多尺度预测的总loss之和)
            result = model(images)

            for i in range(3):
                resolution = result[i]
                # 对不同解析度下的 feature map 进行解码，得到预测框位置，置信度以及预测类型
                pred_boxes, conf, pred_cls = model_utils.decode(resolution, i)
                # [1, 3, 13, 13, 4] 预测框的形状，其意义为：
                # 对于batch_size里面的每一张图片，在3个不同的分辨率尺度下，对于13*13个网格中，分别预测
                # 一个预测框
                print(pred_boxes.shape)
                anchor_idx = [_ for _ in range(3 * i, 3 * i + 3)]
                gt_num, correct_num, mask, conf_mask, tx, ty, tw, th, tconf, tcls = model_utils.cal_statics(
                    pred_boxes=pred_boxes, labels=targets, origin_anchors=anchors[anchor_idx],
                    feature_shape=feature_shape[i], ignore_threshold=0.5
                )

            # print(targets.shape)

            print("OK!!!")


if __name__ == "__main__":
    main()
