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
from torch.utils import data
from torch.backends import cudnn

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

    dataloader = torch.utils.data.DataLoader(
        txt_loader(dataset_cfg.train_img_list_txt),
        batch_size=dataset_cfg.batch_size, shuffle=False)


if __name__ == "__main__":
    main()
