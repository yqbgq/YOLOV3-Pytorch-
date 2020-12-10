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


from utils.default_config import train_cfg, dataset_cfg, hyper_pars_cfg


def main():
    train_pars = train_cfg              # get some pars relate to training
    dataset_pars = dataset_cfg          # get some pars relate to dataset
    hyper_pars_pars = hyper_pars_cfg    # get some pars relate to hyper_pars

    os.makedirs('../checkpoints', exist_ok=True)  # the folders used to stored the checkpoints

    train_path_txt = dataset_pars.train_img_list_txt    # 训练数据集存储的位置

    learning_rate = hyper_pars_pars.learning_rate
    momentum = hyper_pars_pars.momentum
    decay = hyper_pars_pars.decay
    burn_in = hyper_pars_pars.burn_in


