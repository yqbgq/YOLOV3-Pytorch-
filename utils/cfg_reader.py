# ================================================================
#
#   Editor      : Pycharm
#   File name   : cfg_reader
#   Author      : HuangWei
#   Created date: 2020-12-10 14:12
#   Description : This file help us to read pars
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
import configparser
import os


class cfg:
    """
    我承认这里使用据对路径写的很丑，但是当这个类被导入到其他 py 文件中使用的时候，会出现相对路径错误的问题
    我还没有想到有什么比较好的方法来解决，但是我觉得，可以直接使用配置类，别用 ini 配置文件了
    There is no doubt that my way to get the path of ini is very ugly
    but I have no better to solve the Absolute path and relative path in multi-layered folders
    """
    __base = "C://Users//huangwei//Desktop//项目//myYOLOV3//"
    __path = os.path.join(__base, "yolo_par.ini")
    __cf = configparser.ConfigParser()
    __cf.read(__path)

    @staticmethod
    def get_par(catalogue, key):
        return cfg.__cf.get(catalogue, key)


