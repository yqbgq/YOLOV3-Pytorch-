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
from torchnet import meter
from torch import nn
from tqdm import trange

from utils.default_config import train_cfg, dataset_cfg, hyper_pars_cfg
from utils import model_utils
from model import darknet53
from dataset.load_from_txt import txt_loader

anchors = torch.Tensor(
    [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
)
feature_shape = [52, 26, 13]

lambda_coord = 1
mse_loss = nn.MSELoss()  # 均方误差 损失函数，计算 检测时的坐标损失
bce_loss = nn.BCELoss()  # 计算目标和输出之间的二进制交叉熵  损失函数，计算  多类别的分类损失


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

    for step in range(train_cfg.epochs):
        # 清空仪表信息和混淆矩阵信息
        loss_meter.reset()

        total_obs_num = 0
        total_right_num = 0
        total_loss = 0

        with trange(len(dataloader)) as t:
            iter_data = iter(dataloader)
            # 每轮epoch
            for _ in t:
                (images, targets) = next(iter_data)

                # images :处理后的图像tensor[16,3,416,416]        targets:坐标被归一化后的真值框filled_labels[16,50,5] 值在0-1之间
                if train_cfg.use_cuda:
                    images = images.cuda()
                    targets = targets.cuda()

                # 优化器梯度清零
                optimizer.zero_grad()
                # 得到网络输出值，作为损失 (loss :多尺度预测的总loss之和)
                result = model(images)
                # loss = model(images, targets)
                loss = torch.zeros(1)
                loss = loss.cuda().requires_grad_()  # type: torch.Tensor

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
                        pred_boxes=pred_boxes, labels=targets, origin_anchors=anchors[anchor_idx],
                        feature_shape=feature_shape[i], ignore_threshold=0.5
                    )

                    # 检查预测框对应的置信度大于 0.25 的数量，即有效预测框的数量
                    proposal_num = int((conf > 0.25).sum().item())

                    # 计算召回率
                    total_obs_num += float(gt_num)
                    total_right_num += float(correct_num)
                    # recall = float(correct_num / gt_num) if gt_num else 1
                    recall = float(total_right_num / total_obs_num) if total_obs_num else 1

                    # Handle masks

                    cls_mask = mask.unsqueeze(-1).repeat(1, 1, 1, 1, 80)

                    # Mask outputs to ignore non-existing objects  通过掩码来忽略 不存在物体
                    # mask 初始化全为0，只有  在3个原始锚框与 真值框 iou最大的那个锚框  对应的预测框位置置为1，即  负责检测物体的位置为1
                    loss_x = lambda_coord * bce_loss(x * mask, tx * mask)
                    loss_y = lambda_coord * bce_loss(y * mask, ty * mask)
                    loss_w = (lambda_coord * mse_loss(w * mask, tw * mask) / 2)  # 为何 /2 ?
                    loss_h = (lambda_coord * mse_loss(h * mask, th * mask) / 2)
                    # 有无物体损失  conf_mask  [16,3,13,13]  初始化全1，之后的操作：负责预测物体的网格置为1，它周围网格置为0
                    loss_conf = bce_loss(conf * conf_mask, tconf * conf_mask)
                    # 多分类损失
                    loss_cls = bce_loss(pred_cls * cls_mask, tcls * cls_mask)
                    loss = loss + loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
                    # 总loss为 loss_x、loss_y、loss_w、loss_h、loss_conf、loss_cls之和

                total_loss += float(loss)

                if _ % 4 == 0:
                    t.set_description(  # 修改表头显示，增加时间和epoch显示
                        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) +
                        " Epoch {}".format(step)
                    )

                    t.set_postfix_str("Loss: {:.4f}".format(float(total_loss / (_+1)))
                                      + " Recall: {:.4f}%".format(recall * 100))

                if _ % 50 == 0:
                    with open("../record.txt", "a+") as f:
                        f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) +
                                " Epoch {}".format(step))
                        f.write(" Loss: {:.4f}".format(float(total_loss / (_+1)))
                                + " Recall: {:.4f}%".format(recall * 100))
                        f.write("\n")

                loss.backward()
                # print(model.conv_set_1.conv_layer_0.conv_layer.weight)
                # 更新优化器的可学习参数
                optimizer.step()
                # print(model.conv_set_1.conv_layer_0.conv_layer.weight)
                loss_meter.add(loss.item())

        if step % 1 == 0:
            checkpoint = {'model': model.state_dict()}
            torch.save(checkpoint, train_cfg.checkpoint_dir + '/' + str(step) + 'yolov3.pt')
        print("第" + str(step) + "次epoch完成==========================")


if __name__ == "__main__":
    main()
