# ================================================================
#   Copyright, All Rights Reserved
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


