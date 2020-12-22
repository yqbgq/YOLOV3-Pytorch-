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
import time

import torch
from torch import optim
from torch.utils import data
from torch.backends import cudnn
from torch import nn
from tqdm import trange
import argparse
import sys
import numpy as np

sys.path.append('..')

from utils.default_config import train_cfg, dataset_cfg, hyper_pars_cfg
from utils import model_utils
from model import darknet53
from dataset.load_from_txt import txt_loader


class yolo:
    def __init__(self, step_count):
        self.step = step_count
        self.anchors = torch.Tensor(
            [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
        )
        self.feature_shape = [52, 26, 13]
        self.lambda_coord = 1
        self.mse_loss = nn.MSELoss()  # 均方误差 损失函数，计算 检测时的坐标损失
        self.bce_loss = nn.BCELoss()  # 计算目标和输出之间的二进制交叉熵  损失函数，计算  多类别的分类损失

        self.train_path_txt = dataset_cfg.train_img_list_txt  # 训练数据集存储的位置

        os.makedirs('../checkpoints', exist_ok=True)  # the folders used to stored the checkpoints

        self.__init_model()
        if self.step > 0:
            self.__load_weight()

    def __init_model(self):
        self.learning_rate = hyper_pars_cfg.learning_rate
        self.momentum = hyper_pars_cfg.momentum
        self.decay = hyper_pars_cfg.decay
        self.burn_in = hyper_pars_cfg.burn_in

        self.model = darknet53.DarkNet()

    def __init_train(self):
        self.model.apply(model_utils.init_weight)  # 对模型参数进行初始化
        if train_cfg.use_cuda:
            self.model = self.model.cuda()
            cudnn.benchmark = True
            cudnn.enabled = True

        # 数据集加载器
        self.dataloader = torch.utils.data.DataLoader(
            txt_loader(self.train_path_txt),
            batch_size=train_cfg.batch_size, shuffle=False)

        # 定义了一个优化器，真正的大佬都使用 SGD，只有萌新采用 Adam，手动狗头
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, dampening=0,
                                   weight_decay=self.decay)

    def __cal_loss(self, images, targets):
        self.optimizer.zero_grad()
        # 得到网络输出值，作为损失 (loss :多尺度预测的总loss之和)
        result = self.model(images)
        # loss = model(images, targets)
        loss = torch.zeros(1).cuda()    # type: torch.Tensor

        temp_gt_num = 0
        temp_right_num = 0

        for i in range(3):
            resolution = result[i]
            # 对不同解析度下的 feature map 进行解码，得到预测框位置，置信度以及预测类型
            pred_boxes, conf, pred_cls, x, y, w, h = model_utils.decode(resolution, i)
            # [1, 3, 13, 13, 4] 预测框的形状，其意义为：
            # 对于batch_size里面的每一张图片，在3个不同的分辨率尺度下，对于13*13个网格中，分别预测
            # 一个预测框
            # print(pred_boxes.shape)
            anchor_idx = [_ for _ in range(3 * i, 3 * i + 3)]

            gt_num, correct_num, mask, conf_mask, tx, ty, tw, th, tconf, tcls = model_utils.cal_statics(
                pred_boxes=pred_boxes, labels=targets, origin_anchors=self.anchors[anchor_idx],
                feature_shape=self.feature_shape[i], ignore_threshold=0.5
            )

            # 检查预测框对应的置信度大于 0.25 的数量，即有效预测框的数量
            # proposal_num = int((conf > 0.25).sum().item())

            # 计算召回率
            temp_gt_num += float(gt_num)
            temp_right_num += float(correct_num)
            # recall = float(correct_num / gt_num) if gt_num else 1
            # recall = float(total_right_num / total_obs_num) if total_obs_num else 1

            # Handle masks

            cls_mask = mask.unsqueeze(-1).repeat(1, 1, 1, 1, 80)

            # Mask outputs to ignore non-existing objects  通过掩码来忽略 不存在物体
            # mask 初始化全为0，只有  在3个原始锚框与 真值框 iou最大的那个锚框  对应的预测框位置置为1，即  负责检测物体的位置为1
            loss_x = self.lambda_coord * self.bce_loss(x * mask, tx * mask)
            loss_y = self.lambda_coord * self.bce_loss(y * mask, ty * mask)
            loss_w = (self.lambda_coord * self.mse_loss(w * mask, tw * mask) / 2)  # 为何 /2 ?
            loss_h = (self.lambda_coord * self.mse_loss(h * mask, th * mask) / 2)
            # 有无物体损失  conf_mask  [16,3,13,13]  初始化全1，之后的操作：负责预测物体的网格置为1，它周围网格置为0
            loss_conf = self.bce_loss(conf * conf_mask, tconf * conf_mask)
            # 多分类损失
            loss_cls = self.bce_loss(pred_cls * cls_mask, tcls * cls_mask)
            loss = loss + loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            # 总loss为 loss_x、loss_y、loss_w、loss_h、loss_conf、loss_cls之和

        return loss, temp_gt_num, temp_right_num

    def __load_weight(self):
        path = train_cfg.checkpoint_dir + '/' + str(self.step - 1) + 'yolov3.pt'
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])

    def train(self):
        self.__init_train()
        self.model.train()

        total_obs_num = 0
        total_right_num = 0
        total_loss = 0
        with trange(len(self.dataloader)) as t:
            iter_data = iter(self.dataloader)
            for batch in t:
                (images, targets) = next(iter_data)

                # images :处理后的图像tensor[16,3,416,416]        targets:坐标被归一化后的真值框filled_labels[16,50,5] 值在0-1之间
                if train_cfg.use_cuda:
                    images = images.cuda()
                    targets = targets.cuda()

                loss, temp_gt_num, temp_right_num = self.__cal_loss(images, targets)

                total_obs_num += temp_gt_num
                total_right_num += temp_right_num
                total_loss += float(loss)

                recall = float(total_right_num / total_obs_num) if total_obs_num else 1

                des = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + " Epoch {}".format(self.step)
                post = "Loss: {:.4f}".format(total_loss / (batch + 1)) + " Recall: {:.4f}%".format(
                    recall * 100)

                # 修改表头显示，增加时间和epoch显示
                t.set_description(des)
                t.set_postfix_str(post)

                if batch % 50 == 0:
                    with open("../record.txt", "a+") as f:
                        f.write(des + " " + post + "\n")

                loss.backward()
                # 更新优化器的可学习参数
                self.optimizer.step()
            checkpoint = {'model': self.model.state_dict()}
            torch.save(checkpoint, train_cfg.checkpoint_dir + '/' + str(self.step) + 'yolov3.pt')
            print("第" + str(self.step) + "次epoch完成==========================")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--step', type=int, default=64, metavar='N')
    # args = parser.parse_args()
    #
    # step = args.step

    for step in range(30):
        yolo_instance = yolo(step)
        yolo_instance.train()
