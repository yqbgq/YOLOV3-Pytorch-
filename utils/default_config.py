# ================================================================
#   Copyright, All Rights Reserved
#
#   Editor      : Pycharm
#   File name   : default_config
#   Author      : HuangWei
#   Created date: 2020-12-10 14:55
#   Email       : 446296992@qq.com
#   Description : this class hold some default pars
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================

class train_config:
    epochs = 30  # 训练轮数
    image_folder = 'data/samples'  # 数据集地址
    batch_size = 16  # batch大小
    # model_config_path = 'config/yolov3.cfg'  # 模型网络结构
    # data_config_path = 'config/coco.data'  # 配置数据集的使用情况
    class_path = 'data/coco.names'  # coco数据集类别标签
    conf_thres = 0.8  # 物体置信度阈值
    nms_thres = 0.4  # iou for nms的阈值
    # n_cpu = 0  # 批生成期间要使用的cpu线程数
    img_size = 416  # 输入图像尺寸的大小
    use_cuda = True  # 是否使用GPU
    # visdom = True  # 是否使用visdom来可视化loss
    print_freq = 8  # 训练时，每N个batch显示
    lr_decay = 0.1  # 1e-3 -> 1e-4

    checkpoint_interval = 1  # 每隔几个模型保存一次
    checkpoint_dir = './checkpoints'  # 保存生成模型的路径

    # load_model_path = None  # 加载预训练的模型的路径，为None代表不加载
    # load_model_path=checkpoint_dir+'/latestbobo.pt'  # 预训练权重   (仅.pt)


# 初始化该类的一个对象
train_cfg = train_config()


