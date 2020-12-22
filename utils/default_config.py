# ================================================================
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
    batch_size = 2  # batch大小
    # model_config_path = 'config/yolov3.cfg'  # 模型网络结构
    # data_config_path = 'config/coco.data'  # 配置数据集的使用情况
    class_path = 'data/coco.names'  # coco数据集类别标签
    conf_threshold = 0.8  # 物体置信度阈值
    nms_threshold = 0.4  # iou for nms的阈值
    # n_cpu = 0  # 批生成期间要使用的cpu线程数
    img_size = 416  # 输入图像尺寸的大小
    use_cuda = True  # 是否使用GPU
    print_freq = 8  # 训练时，每N个batch显示
    lr_decay = 0.1  # 1e-3 -> 1e-4

    checkpoint_interval = 1  # 每隔几个模型保存一次
    checkpoint_dir = '../checkpoints'  # 保存生成模型的路径


class dataset_config:
    classes = 80
    train_img_list_txt = "../data/coco/trainvalno5k.txt"
    valid = "data/coco/5k.txt"
    names = "data/coco.names"
    # 使用前缀进行拼接，得到绝对地址，感觉这段代码很丑，以后想想办法
    prefix = "C://Users//huangwei//Desktop//PytorchNetHub//Yolov3_pytorch//data//coco"


class hyper_pars_config:
    batch = 16
    subdivisions = 1
    width = 416
    height = 416
    channels = 3
    momentum = 0.9
    decay = 0.0005
    angle = 0
    saturation = 1.5
    exposure = 1.5
    hue = .1

    learning_rate = 0.001
    burn_in = 1000
    max_batches = 500200
    # policy = steps
    steps = 400000, 450000
    scales = .1, .1


# 初始化该类的一个对象
train_cfg = train_config()
dataset_cfg = dataset_config()
hyper_pars_cfg = hyper_pars_config()
