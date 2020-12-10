# ================================================================
#   Copyright, All Rights Reserved
#
#   Editor      : Pycharm
#   File name   : test_cfg_reader
#   Author      : HuangWei
#   Created date: 2020-12-10 14:14
#   Description : 测试配置读取函数
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================

from utils.cfg_reader import cfg

print(cfg.get_par("test", "value1"))
