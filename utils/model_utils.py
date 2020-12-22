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
import numpy as np
import math

from utils.default_config import dataset_cfg

anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]

strides = [8, 16, 32]


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def cal_statics(pred_boxes, labels: torch.Tensor, origin_anchors: torch.Tensor, feature_shape, ignore_threshold):
    batch_size = labels.shape[0]  # Batch 中图像的数量
    anchors_num = origin_anchors.shape[0]  # 锚框的个数 为 3
    class_num = 80  # 预测的类别数量
    img_shape = feature_shape  # 特征图的大小 13 26 52

    # 初始化参数
    conf_mask = torch.ones(batch_size, anchors_num, img_shape, img_shape).cuda()  # [16,3,13,13]   全1
    mask = torch.zeros(batch_size, anchors_num, img_shape, img_shape).cuda()  # [16,3,13,13]   全0
    tx = torch.zeros(batch_size, anchors_num, img_shape, img_shape).cuda()  # [16,3,13,13]   全0
    ty = torch.zeros(batch_size, anchors_num, img_shape, img_shape).cuda()  # [16,3,13,13]   全0
    tw = torch.zeros(batch_size, anchors_num, img_shape, img_shape).cuda()  # [16,3,13,13]   全0
    th = torch.zeros(batch_size, anchors_num, img_shape, img_shape).cuda()  # [16,3,13,13]   全0
    tconf = torch.zeros(batch_size, anchors_num, img_shape, img_shape).cuda()  # [16,3,13,13]   全0
    tcls = torch.zeros(batch_size, anchors_num, img_shape, img_shape, class_num).cuda()  # [16,3,13,
    # 13,80]  全0

    gt_num = 0  # 真值的数量，用于计算召回率
    correct_num = 0  # 预测出有物体的个数，即真值框与原始锚框IOU最大的那个锚框对应的预测框， IOU>0.5则预测正确

    for i in range(batch_size):
        # 需要注意，这里的labels已经被填充为 [batch_size, 50, 5]
        for j in range(labels.shape[1]):
            # 表示已经遍历完所有的物体真值框
            if torch.sum(labels[i, j]) == 0:
                break

            gt_num += 1

            # 标签值中的坐标都经过了归一化，处于 0-1 之间
            gt_x = labels[i, j, 1] * img_shape  # 转化为在特征图上的坐标
            gt_y = labels[i, j, 2] * img_shape
            gt_w = labels[i, j, 3] * img_shape
            gt_h = labels[i, j, 4] * img_shape

            # 获取网格框的索引，即左上角的角标
            gt_i = int(gt_x)
            gt_j = int(gt_y)

            # gt_box 的预测框的shape为 [1,4]
            gt_box = torch.FloatTensor([0, 0, gt_w, gt_h]).unsqueeze(0)

            # Get shape of anchor box [3,4]   前两列全为0  后两列为 三个anchor的w、h
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(origin_anchors), 2)),
                                                              np.array(origin_anchors)), 1))

            # Calculate iou between gt and anchor shapes
            # 计算 一个真值框 与  对应的3个原始锚框  之间的iou
            anchors_iou = bbox_iou(gt_box, anchor_shapes)

            # Where the overlap is larger than threshold set mask to zero (ignore)   当iou重叠率>阈值，则置为0
            # conf_mask全为1 [16,3,13,13]  当一个真值框 与  一个原始锚框  之间的iou > 阈值时，则置为0。
            # 即 将 负责预测物体的网格及 它周围的网格 都置为0 不参与训练，后面的代码会 将负责预测物体的网格再置为1。
            conf_mask[i, anchors_iou > ignore_threshold] = 0

            # Find the best matching anchor box  找到 一个真值框 与  对应的3个原始锚框  之间的iou最大的  下标值
            best_n = np.argmax(anchors_iou)

            # Get ground truth box [1,4]
            gt_box = torch.FloatTensor([gt_x, gt_y, gt_w, gt_h]).unsqueeze(0).cuda().requires_grad_()

            # Get the best prediction  [1,4]
            # pred_boxes:在13x13尺度上的预测框
            # pred_box：取出  3个原始锚框与 真值框 iou最大的那个锚框  对应的预测框
            # 注意这里是 gt_j， gt_i的格式，因为 gt_j是对y的向下取整
            pred_box = pred_boxes[i, best_n, gt_j, gt_i].unsqueeze(0)

            # Masks   [16,3,13,13]   全0      在3个原始锚框与 真值框 iou最大的那个锚框  对应的预测框位，
            # 即 负责预测物体的网格置为1 （此时它周围网格为0，思想类似nms）
            mask[i, best_n, gt_j, gt_i] = 1

            #  [16,3,13,13]   全1 然后将 负责预测物体的网格及 它周围的网格 都置为0 不参与训练 ，然后  将负责预测物体的网格再次置为1。
            #  即总体思想为： 负责预测物体的网格 位置置为1，它周围的网格置为0。类似NMS 非极大值抑制
            conf_mask[i, best_n, gt_j, gt_i] = 1

            # Coordinates 坐标     gi= gx的向下取整。  gx-gi、gy-gj 为 网格内的 物体中心点坐标（0-1之间）
            # tx  ty初始化全为0，在有真值框的网格位置写入 真实的物体中心点坐标
            tx[i, best_n, gt_j, gt_i] = gt_x - gt_i
            ty[i, best_n, gt_j, gt_i] = gt_y - gt_j

            # Width and height
            #  论文中 13x13尺度下真值框=原始锚框 x 以e为底的 预测值。故预测值= log(13x13尺度下真值框  / 原始锚框  +  1e-16 )
            tw[i, best_n, gt_j, gt_i] = math.log(gt_w / anchors[best_n.data][0] + 1e-16)
            th[i, best_n, gt_j, gt_i] = math.log(gt_h / anchors[best_n.data][1] + 1e-16)

            # One-hot encoding of label
            tcls[i, best_n, gt_j, gt_i, int(labels[i, j, 0])] = 1

            # Calculate iou between ground truth and best matching prediction 计算真值框 与
            # 3个原始锚框与真值框iou最大的那个锚框对应的预测框
            # 之间的iou
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            # [16,3,13,13]   全0，有真值框对应的网格位置为1  标明 物体中心点落在该网格中，该网格去负责预测物体
            tconf[i, best_n, gt_j, gt_i] = 1

            if iou > 0.5:
                correct_num += 1
    return gt_num, correct_num, mask, conf_mask, tx, ty, tw, th, tconf, tcls


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
    :param i: 是哪一种分辨率 0， 1 ，2逐渐减小
    :return: 网格
    """
    # shape: [batch_size, 255, feature_map_shape, feature_map_shape]
    conv_output = conv_output
    conv_shape = conv_output.shape
    batch_size = conv_shape[0]
    output_size = conv_shape[2]

    stride = 416 / output_size
    mask = [_ for _ in range(i * 3, i * 3 + 3)]

    # 将网络输出的结果转换为 [batch_size, 3, img_shape, img_shape, 85]
    conv_output = torch.reshape(conv_output, [batch_size, 3, output_size, output_size, 5 + dataset_cfg.classes])

    # Get outputs    85中0-3 为预测的框偏移，4为 物体置信度（是否有物体）  5： 为多类别的分类概率
    x = torch.sigmoid(conv_output[..., 0])  # Center x  [16,3,13,13]
    y = torch.sigmoid(conv_output[..., 1])  # Center y  [16,3,13,13]
    w = conv_output[..., 2]  # Width     [16,3,13,13]
    h = conv_output[..., 3]  # Height    [16,3,13,13]
    conf = torch.sigmoid(conv_output[..., 4])  # Conf      [16,3,13,13]
    pred_cls = torch.sigmoid(conv_output[..., 5:])  # Cls pred. [16,3,13,13,80]

    # 好了，接下来需要画网格了，其中，output_size 等于 13、26 或者 52

    # Calculate offsets for each grid 计算每个网格的偏移量
    # torch.linspace返回 start 和 end 之间等间隔 steps 点的一维 Tensor
    # repeat沿着指定的尺寸重复 tensor
    # 过程：
    #      torch.linspace(0, g_dim-1, g_dim)  ->  [1,13]的tensor
    #      repeat(g_dim,1)                    ->  [13,13]的tensor 每行内容为0-12,共13行
    #      repeat(bs*self.num_anchors, 1, 1)  ->  [48,13,13]的tensor   [13,13]内容不变，在扩展的一维上重复48次
    #      view(x.shape)                      ->  resize成[16.3.13.13]的tensor
    # grid_x、grid_y用于 定位 feature map的网格左上角坐标
    grid_x = torch.linspace(0, output_size - 1, output_size).repeat(output_size, 1).repeat(
        batch_size * 3, 1, 1).view(x.shape).cuda()  # [16.3.13.13]  每行内容为0-12,共13行

    grid_y = torch.linspace(0, output_size - 1, output_size).repeat(output_size, 1).t().repeat(
        batch_size * 3, 1, 1).view(y.shape).cuda()  # [16.3.13.13]  每列内容为0-12,共13列（因为使用转置T）

    scaled_anchors = torch.Tensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])  # 将 原图尺度的锚框也缩放到统一尺度下

    anchor_w = scaled_anchors[:, 0][mask].cuda()
    anchor_h = scaled_anchors[:, 1][mask].cuda()

    # 这里要判断一下是不是有 CUDA 然后转换到 GPU 上
    anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, output_size * output_size).view(h.shape)
    anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, output_size * output_size).view(w.shape)

    # Add offset and scale with anchors  给锚框添加偏移量和比例
    pred_boxes = torch.Tensor(conv_output[..., :4].shape).cuda().requires_grad_()  # 新建一个tensor[16,3,13,13,4]
    # pred_boxes为 在13x13的feature map尺度上的预测框
    # x,y为预测值（网格内的坐标，经过sigmoid之后值为0-1之间） grid_x，grid_y定位网格左上角偏移坐标（值在0-12之间）
    pred_boxes[..., 0] = x.data + grid_x
    pred_boxes[..., 1] = y.data + grid_y
    # w，h为 预测值，即相对于原锚框的偏移值    anchor_w，anchor_h为 网格对应的3个锚框
    pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
    pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

    return pred_boxes, conf, pred_cls, x, y, w, h
